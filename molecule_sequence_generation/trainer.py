import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import logging
import numpy as np
from scipy.stats import pearsonr
import wandb
import math
import os

logger = logging.getLogger("trainer")

# =======================================
# Helper Functions
# =======================================
def get_gpt_like_lr_scheduler(optimizer, warmup_steps, total_steps, min_lr_ratio=0.1):
    """
    Creates a learning rate scheduler with a linear warmup followed by cosine decay.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer to adjust.
        warmup_steps (int): Number of steps for linear warmup.
        total_steps (int): Total number of training steps.
        min_lr_ratio (float): The minimum learning rate as a fraction of the initial learning rate.

    Returns:
        scheduler (LambdaLR): A PyTorch LambdaLR scheduler.
    """

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Linear warmup: from 0 to 1.
            return float(current_step) / float(max(1, warmup_steps))
        else:
            # Cosine decay.
            progress = float(current_step - warmup_steps) / float(
                max(1, total_steps - warmup_steps)
            )
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            # Scale down to min_lr_ratio at the end of training.
            decayed = (1 - min_lr_ratio) * cosine_decay + min_lr_ratio
            return decayed

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return scheduler


# =======================================
# Dataset Class
# =======================================
class ClassifierDataset(Dataset):
    def __init__(self, x=None, y=None, loss_masks=None):
        self.x = x
        self.y = y
        self.loss_masks = loss_masks

    def __len__(self):
        return len(self.x) if self.x is not None else 0

    def __getitem__(self, idx):
        # Return three items: features, targets, and loss masks.
        return self.x[idx], self.y[idx], self.loss_masks[idx]

    def add_data(self, x, y, loss_masks):
        if self.x is None:
            self.x = x
            self.y = y
            self.loss_masks = loss_masks
        else:
            self.x = torch.cat([self.x, x], dim=0)
            self.y = torch.cat([self.y, y], dim=0)
            self.loss_masks = torch.cat([self.loss_masks, loss_masks], dim=0)

    def collate_fn(self, batch):
        x = torch.cat([item[0] for item in batch], dim=0)
        y = torch.cat([item[1] for item in batch], dim=0)
        loss_masks = torch.cat([item[2] for item in batch], dim=0)
        return x, y, loss_masks

    def save_dataset(self, path):
        dataset_dict = {"x": self.x, "y": self.y, "loss_masks": self.loss_masks}
        torch.save(dataset_dict, path)

    def load_dataset(self, path):
        dataset_dict = torch.load(path)
        self.x = dataset_dict["x"]
        self.y = dataset_dict["y"]
        self.loss_masks = dataset_dict["loss_masks"]



# =======================================
# Trainer Class
# =======================================
class Trainer:

    def log_reward(self, config, reward_mean, step, folder="eval"):
        if self.config.wandb and self.config.rank == 0:
            wandb.log({f"{folder}/{config}": reward_mean, f"{folder}/epoch": step})

    def reward_model_wrapper(self, x):
        self.raw_model.head.eval()
        self.raw_model.embedding.eval()
        self.raw_model.head.to(x.device)
        self.raw_model.embedding.to(x.device)
        return self.raw_model.head(self.raw_model.embedding(x.float())).squeeze(2).detach()

    def __init__(self, model, config):
        """
        Args:
            model: A torch.nn.Module that must implement an embedding and head.
            config: A configuration object with attributes:
                - config.model: holds model-specific parameters (e.g., bucketing flag, etc.)
                - config.train: holds training parameters (e.g., learning_rate, num_epochs, batch_size, grad_accumulation_steps, grad_norm_clip, log_interval, reinit_classifier)
                - config.world_size: total number of processes for distributed training.
                - config.rank: the rank (or GPU id) for this process.
        """
        self.model = model
        self.config = config

        self.model_config = config.model
        self.train_config = config.train
        self.best_performance = 0
        self.is_setup = False

    def setup(self):
        # Distributed setup: if more than one process, wrap the model in DistributedDataParallel.
        # Set device
        self.device = torch.device("cuda", self.config.rank)
        self.model = self.model.to(self.device)
        self.model.ref_model = self.model.ref_model.to(self.device)

        if self.config.world_size > 1:
            dist.init_process_group(backend="nccl")
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[self.device]
            )
        if self.config.world_size > 1:
            self.raw_model = self.model.module
        else:
            self.raw_model = self.model
        # Set optimizer
        self.optimizer = self.raw_model.configure_optimizers(
            self.train_config.learning_rate,
            self.train_config.weight_decay,
            self.train_config.betas,
        )

        if self.train_config.lr_decay:
            self.scheduler = get_gpt_like_lr_scheduler(
                self.optimizer,
                warmup_steps=50,
                total_steps=150,
                min_lr_ratio=0,
            )

        # Initialize training and validation datasets
        self.train_classifier_dataset = ClassifierDataset()
        self.val_classifier_dataset = ClassifierDataset()
        # Optionally save initial classifier state if reinitialization is desired
        if self.train_config.reinit_classifier:
            self.initialized_head_state_dict = self.raw_model.head.state_dict().copy()
            self.initialized_embedding_state_dict = (
                self.raw_model.embedding.state_dict().copy()
            )
        # Set up gradient scaler for AMP
        self.scaler = torch.cuda.amp.GradScaler()
        self.loaded_checkpoint = False

        self.is_setup = True

    def compute_rewards(self, epoch):
        self.model.eval()
        if self.train_config.task == "rna":
            self.model.embedding.gru_tower.gru.train()

        for guidance_scale in [10, 50]:
            y_mean_full_trajectory = self.generate_samples_full_trajectory(
                use_classifier=True,
                collapse_to_mean=self.model_config.bucketing,
                eta=3,
                guidance_scale=guidance_scale,
                iterations=1,
            )
            
            logger.info(
                "==> Collapse to Mean: %s, Epoch %d Reward Mean: %f", self.model_config.collapse_to_mean, epoch, y_mean_full_trajectory
            )
            
            self.log_reward(f"guidance_scale_{guidance_scale}" + ("_collapse_to_mean" if self.model_config.bucketing else ""), y_mean_full_trajectory, epoch, folder="eval")
            
            if y_mean_full_trajectory > self.best_performance:
                # remove previous best checkpoint
                if os.path.exists(f"/scratch/result_rna/best_checkpoint_dna_{self.best_performance}.pt"):
                    os.remove(f"/scratch/result_rna/best_checkpoint_dna_{self.best_performance}.pt")
                self.best_performance = y_mean_full_trajectory
                self.save_checkpoint(f"/scratch/result_rna/best_checkpoint_dna_{y_mean_full_trajectory}.pt")

        self.model.train()

    def fit(self):
        assert self.is_setup, "Trainer not setup! Please call setup() first."

        max_eta = 4 if self.model_config.bucketing else None

        # Start the training loop over epochs
        for epoch in range(self.train_config.num_epochs):

            if epoch > 0:
                self.compute_rewards(epoch)
            
            if self.config.exit_after_first_epoch and epoch > 0:
                break
            
            if self.train_config.reset_dataset:
                self.train_classifier_dataset = ClassifierDataset()
                self.val_classifier_dataset = ClassifierDataset()

            # Uncomment this if you want to load a dataset from a previous run, this was helpful for debugging / speeding up training if you want to start from a previous run
            # if epoch == 0:
            #     self.train_classifier_dataset.load_dataset(
            #         f"train_classifier_dataset_21.pt"
            #     )
            #     self.val_classifier_dataset.load_dataset(
            #         f"val_classifier_dataset_21.pt"
            #     )
            # else:
            self.collect_data(iteration=epoch, max_eta=max_eta)

            if epoch > 0:
                self.train_classifier_dataset.save_dataset(f"/scratch/result_rna/train_classifier_dataset_{epoch}.pt")
                self.val_classifier_dataset.save_dataset(f"/scratch/result_rna/val_classifier_dataset_{epoch}.pt")
                print("saved datset")

            # if reinit classifier, reset head and embedding
            if self.train_config.reinit_classifier:
                self.raw_model.head.load_state_dict(self.initialized_head_state_dict)
                self.raw_model.embedding.load_state_dict(
                    self.initialized_embedding_state_dict
                )
                self.optimizer = self.raw_model.configure_optimizers(
                    self.train_config.learning_rate,
                    self.train_config.weight_decay,
                    self.train_config.betas,
                )
                if self.train_config.lr_decay:
                    self.scheduler = get_gpt_like_lr_scheduler(
                        self.optimizer,
                        warmup_steps=50,
                        total_steps=150,
                        min_lr_ratio=0,
                    )
                    
            train_avg_loss = self.epoch(epoch_idx=epoch)
            test_avg_loss, test_pearson_corr, test_explained_variance = self.evaluate(
                max_eta=max_eta
            )

            if self.config.wandb and self.config.rank == 0:
                wandb.log(
                    {
                        "train/loss": train_avg_loss,
                        "test/loss": test_avg_loss,
                        "test/pearson": test_pearson_corr,
                        "test/explained_variance": test_explained_variance,
                    }
                )

    def collect_data(self, iteration, max_eta=None):
        """
        Collects data for training and validation using guided sample generation.

        For training samples:
          - If iteration == 0, samples are generated without classifier guidance.
          - Otherwise, classifier-guided sampling is used (with the provided max_eta).

        For validation samples:
          - If iteration > 0, classifier-guided sampling is used.
          - Otherwise, samples are generated without classifier guidance.

        The generated samples include features (x), targets (y), and loss masks.
        These new samples are appended to the training and validation datasets.

        Args:
            iteration (int): The current overall iteration.
            max_eta (float, optional): The best eta value from guided evaluation (used when classifier guidance is on).
        """
        # --- COLLECT TRAINING SAMPLES ---
        if iteration == 0 and not self.loaded_checkpoint:
            logger.info("Collecting initial training samples (without classifier)...")
            use_classifier = False
            eta = None
        else:
            logger.info("Collecting additional training samples (with classifier)...")
            use_classifier = True
            eta = max_eta
    
        # Generate training samples.
        x_train, y_train, loss_masks_train, y_mean_train = (
            self.generate_and_eval_samples(
                use_classifier=use_classifier,
                eta=eta,
                guidance_scale=self.train_config.guidance_scale,
                iterations=self.train_config.train_iterations if iteration > 0 else self.train_config.initial_train_iterations,
            )
        )
        # Append new training samples to the training dataset.
        self.train_classifier_dataset.add_data(x_train, y_train, loss_masks_train)

        # --- COLLECT VALIDATION (TEST) SAMPLES ---
        logger.info("Collecting test samples for classifier evaluation...")

        x_val, y_val, loss_masks_val, _ = self.generate_and_eval_samples(
            use_classifier=use_classifier,
            eta=eta,
            guidance_scale=self.train_config.guidance_scale,
            iterations=self.train_config.classifier_test_iterations,
        )

        # Append new validation samples to the validation dataset.
        self.val_classifier_dataset.add_data(x_val, y_val, loss_masks_val)
        return y_mean_train

    def epoch(self, epoch_idx):
        logger.info(f"Epoch {epoch_idx} training...")
        self.model.train()

        for iteration in range(self.train_config.num_classifier_epochs):

            train_loader = DataLoader(
                self.train_classifier_dataset,
                batch_size=self.train_config.classifier_batch_size,
                shuffle=True,
                drop_last=True,
                collate_fn=self.train_classifier_dataset.collate_fn,
            )

            train_losses = []
            for i, (x, y, loss_masks) in enumerate(train_loader):
                x = x.to(self.device)
                y = y.to(self.device)
                loss_masks = loss_masks.to(self.device)

                loss = self.model(x, y)
                loss = loss[loss_masks == 1]
                loss = loss.mean()

                self.scaler.scale(loss).backward()

                if i % self.train_config.grad_accumulation_steps == 0:
                    # Optionally clip gradients.
                    if self.train_config.grad_norm_clip > 0:
                        self.scaler.unscale_(self.optimizer)
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.train_config.grad_norm_clip
                        )

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.model.zero_grad()
                    if self.train_config.lr_decay:
                        self.scheduler.step()
                train_losses.append(loss.item())

                # Log training progress.
                if i % self.train_config.log_interval == 0:
                    current_lr = self.optimizer.param_groups[0]["lr"]

                    log_dict = {
                        "train/batch_loss": loss.item(),
                        "train/learning_rate": current_lr,
                    }
                    if self.train_config.grad_norm_clip > 0:
                        log_dict["train/grad_norm"] = grad_norm.item()

                    logger.info(
                        f"Epoch {epoch_idx} Iteration {iteration} Batch {i} Loss {loss.item()} LR {current_lr} Grad Norm {grad_norm.item() if self.train_config.grad_norm_clip > 0 else 'N/A'}"
                    )
                    if self.config.wandb and self.config.rank == 0:
                        wandb.log(log_dict)

                if i % self.train_config.eval_interval == 0:
                    test_avg_loss, test_pearson_corr, test_explained_variance = (
                        self.evaluate(max_eta=None)
                    )
                    if self.config.wandb and self.config.rank == 0:
                        wandb.log(
                            {
                                f"test/loss{epoch_idx}": test_avg_loss,
                                f"test/pearson{epoch_idx}": test_pearson_corr,
                                f"test/explained_variance{epoch_idx}": test_explained_variance,
                            }
                        )
                if i % self.train_config.compute_rewards_interval == 0:
                    self.compute_rewards(epoch_idx)
                    self.model.train()

                
        return torch.tensor(train_losses).mean()

    def evaluate(self, max_eta=None):
        """
        Evaluate the classifier on the validation dataset.
        This function creates a DataLoader from the val_classifier_dataset,
        computes the average loss, Pearson correlation, and explained variance over the validation data.
        """
        self.model.eval()
        if self.train_config.task == "rna":
            self.model.embedding.gru_tower.gru.train()
        val_loader = DataLoader(
            self.val_classifier_dataset,
            batch_size=self.train_config.classifier_batch_size,
            shuffle=False,
            collate_fn=self.val_classifier_dataset.collate_fn,
        )
        losses = []
        preds_list = []
        targets_list = []

        with torch.no_grad():
            for i, (x, y, loss_masks) in enumerate(
                tqdm(
                    val_loader,
                    desc="Evaluating",
                    leave=False,
                    disable=(self.config.rank != 0),
                )
            ):
                x = x.to(self.device)
                y = y.to(self.device)
                loss_masks = loss_masks.to(self.device)

                pred = self.raw_model.head(self.raw_model.embedding(x.float()))
                pred = pred.squeeze()

                if self.model_config.bucketing:
                    if self.train_config.task == "dna":
                        y = (50 / 9) * y + (50 / 9)
                        y = y.long()
                        one_hot_encoded = torch.nn.functional.one_hot(y, num_classes=51)
                        one_hot_encoded = one_hot_encoded.squeeze()
                        pred = pred.squeeze()
                        loss = self.raw_model.loss_fct(pred, one_hot_encoded.float())
                    else:
                        y = torch.clamp(y, min=-2, max=2)
                        y = 12.5 * y + 25
                        y = y.long()
                        loss = self.raw_model.loss_fct(pred, y.squeeze())
                else:
                    loss = self.raw_model.loss_fct(pred, y)

                # Apply loss mask and average the loss.
                loss = loss[loss_masks == 1]
                loss = loss.mean()
                losses.append(loss.item())

                # Collect predictions and targets where the mask is active.
                preds_list.append(pred[loss_masks == 1].cpu())
                targets_list.append(y[loss_masks == 1].cpu())

        avg_loss = np.mean(losses) if losses else 0.0

        preds_cat = torch.cat(preds_list, dim=0).to("cpu")
        targets_cat = torch.cat(targets_list, dim=0).to("cpu")

        pred_cat = preds_cat
        targets_cat = targets_cat

        # techicnally, we can compute the pearson correlation and explained variance, never got implemented.
        pearson_corr = float("nan")
        explained_variance = float("nan")

        logger.info(
            f"Evaluation: Loss: {avg_loss}, Pearson Correlation: {pearson_corr}, Explained Variance: {explained_variance}"
        )
        self.model.train()

        return avg_loss, pearson_corr, explained_variance

    def generate_samples_full_trajectory(
        self,
        use_classifier,
        eta,
        guidance_scale=7,
        iterations=1,
        collapse_to_mean=False,
    ):
        """
        Uses the classifier the whole time to generate samples, without ever switching off.
        """
        self.model.eval()
        if self.train_config.task == "rna":
            self.model.embedding.gru_tower.gru.train()
        logger.info(
            f"Evaluating {iterations} samples with classifier: {use_classifier}, guidance_scale: {guidance_scale}, eta: {eta} collapse_to_mean: {collapse_to_mean}"
        )

        if use_classifier:
            if self.train_config.method == "svdd":
                print("USING SVDD")
                predictions = self.raw_model.ref_model.controlled_sample_tweedie(self.reward_model_wrapper, eval_sp_size=self.train_config.inference_batch_size, sample_M=self.train_config.svdd_sample_M, options=False, task=self.train_config.task)

                onehot_samples = self.raw_model.ref_model.transform_samples(predictions)
                predictions = self.raw_model.reward_model(
                    onehot_samples.float().transpose(1, 2)
                ).detach()[:, 0]
            
            else:
                predictions = self.raw_model.controlled_decode_classfier(
                    gen_batch_num=iterations,
                    guidance_scale=guidance_scale,
                    eta=eta if self.model_config.bucketing else None,
                    collapse_to_mean=collapse_to_mean,
                )
        else:
            predictions = self.raw_model.ref_model.decode_sample(
                eval_sp_size=self.train_config.inference_batch_size,
            )
            onehot_samples = self.raw_model.ref_model.transform_samples(predictions)
            predictions = self.raw_model.reward_model(
                onehot_samples.float().transpose(1, 2)
            ).detach()[:, 0]

        return predictions.mean()

    def generate_and_eval_samples(
        self, use_classifier, eta, guidance_scale=7, iterations=10
    ):
        """
        Uses the ref_model's guided_sample method to generate samples, transforms them,
        and then obtains a target reward value.
        Returns a batch of samples (x0) and corresponding targets (y).
        """
        self.model.eval()
        if self.train_config.task == "rna":
            self.model.embedding.gru_tower.gru.train()
        logger.info(
            f"Generating {iterations} samples with classifier: {use_classifier}, guidance_scale: {guidance_scale}, eta: {eta}"
        )
        x0 = []
        y = []
        loss_masks_list = []
        for i in tqdm(
            range(iterations),
            desc="Generating samples",
            disable=(self.config.rank != 0),
        ):

            if self.train_config.method == "svdd":
                samples, mid_samples, loss_masks = (
                    self.raw_model.rollin_rollout_guided_sample_svdd(
                    eval_sp_size=self.train_config.inference_batch_size,
                    use_classifier=use_classifier,
                    guidance_scale=guidance_scale,
                    eta=eta,
                    )
                )
            elif self.train_config.method == "gradient":
                samples, mid_samples, loss_masks = (
                    self.raw_model.rollin_rollout_guided_sample(
                    eval_sp_size=self.train_config.inference_batch_size,
                    use_classifier=use_classifier,
                    guidance_scale=guidance_scale,
                    eta=eta,
                    )
                )
            onehot_samples = self.raw_model.transform_samples(samples)
            target = self.raw_model.reward_model(
                onehot_samples.float().transpose(1, 2)
            ).detach()[:, 0]

            # Convert mid_samples from [127 steps, 36 trajectories, 200] to [36 trajectories, 127 steps, 200]
            mid_samples = torch.stack(mid_samples, dim=0)  # [127, 36, 200]
            mid_samples = mid_samples.transpose(0, 1)  # [36, 127, 200]

            # One-hot encode each step for each trajectory
            batch_size, n_steps, seq_len = mid_samples.shape
            onehot_mid_samples = []
            for i in range(batch_size):
                trajectory_samples = []
                for j in range(n_steps):
                    one_hot = self.raw_model.transform_samples(
                        mid_samples[i, j].unsqueeze(0)
                    )
                    trajectory_samples.append(one_hot)
                trajectory_samples = torch.cat(
                    trajectory_samples, dim=0
                )  # [127, 200, 4]
                onehot_mid_samples.append(trajectory_samples)

            onehot_mid_samples = torch.stack(
                onehot_mid_samples, dim=0
            )  # [36, 127, 200, 4]

            # Add the final samples to the trajectory samples
            onehot_mid_samples = torch.cat(
                [onehot_mid_samples, onehot_samples.unsqueeze(1)], dim=1
            )  # [36, 128, 200, 4]

            # Handle loss masks

            loss_masks = loss_masks.squeeze().to("cpu")
            loss_masks = torch.cat(
                (loss_masks, torch.ones_like(target).to("cpu")), dim=1
            )

            x0.append(onehot_mid_samples)
            y.append(target)
            loss_masks_list.append(loss_masks)

        y = torch.cat(y, dim=0)
        # expand y to match the shape of x0
        y_mean = y.mean(dim=0)  # get last step reward
        y = y.unsqueeze(1).repeat(1, x0[0].shape[1], 1)

        return (
            torch.cat(x0, dim=0).to("cpu"),
            y.to("cpu"),
            torch.cat(loss_masks_list, dim=0).to("cpu"),
            y_mean.to("cpu"),
        )

    def save_checkpoint(self, checkpoint_path):
        torch.save(
            {
                "model_state_dict": self.raw_model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            checkpoint_path,
        )

    def load_checkpoint(self, checkpoint_path):
        self.loaded_checkpoint = True
        logger.info("Loading checkpoint...")
        model_state_dict = torch.load(checkpoint_path)["model_state_dict"]
        self.model.load_state_dict(model_state_dict, strict=True)
