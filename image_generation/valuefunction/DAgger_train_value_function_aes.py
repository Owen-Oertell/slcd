import os
import sys
import numpy as np
import time
import torch
import torch.nn as nn
import torchvision
import random

from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from diffusers import DDIMScheduler
# Accelerate and Logging
from accelerate.logging import get_logger 
import argparse


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


def paste_square(images, grid_size=4):
    """
    images: list of 16 PIL.Image
    return: PIL.Image
    """
    assert len(images) == grid_size ** 2, f"need {grid_size**2} images"

    img_width, img_height = images[0].size
    new_img = Image.new("RGB", (img_width * grid_size, img_height * grid_size))

    for idx, img in enumerate(images):
        row = idx // grid_size
        col = idx % grid_size
        x = col * img_width
        y = row * img_height
        new_img.paste(img, (x, y))
    return new_img
# Set Current Working Directory
cwd = os.getcwd()
sys.path.append(cwd)

# Custom Imports
from aesthetic_scorer import SinusoidalTimeConvNet, AestheticScorerDiff
from vae import  sd_model

import wandb

# Logger
logger = get_logger(__name__)

def parse_aes_args():
    parser = argparse.ArgumentParser(description="Train DAgger Value Function for Aesthetics.")
    # Basic training params
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--latent_dim", type=int, default=4, help="Latent dimension of Stable Diffusion.")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of DDIM inference steps.")
    
    # DAgger specific params
    parser.add_argument("--max_overall_iter", type=int, default=10, help="Maximum DAgger iterations.")
    parser.add_argument("--epoches_per_iter", type=int, default=6, help="Training epochs per DAgger iteration.")
    parser.add_argument("--reinit_classifier", type=lambda x: (str(x).lower() == 'true'), default=True, help="Reinitialize classifier at each DAgger iteration.")
    parser.add_argument("--init_datapoint_num", type=int, default=1500, help="Number of initial data points per class.")
    parser.add_argument("--generation_num", type=int, default=200, help="Number of images to generate per class in DAgger iterations.")

    # Distributional params
    parser.add_argument("--use_buckets", type=lambda x: (str(x).lower() == 'true'), default=True, help="Use distributional output (buckets) for the classifier.")
    parser.add_argument("--num_bins", type=int, default=50, help="Number of bins for distributional output.")
    parser.add_argument("--bin_min", type=float, default=3.0, help="Minimum value for distributional bins (aesthetic score range).")
    parser.add_argument("--bin_max", type=float, default=8.0, help="Maximum value for distributional bins (aesthetic score range).")

    # Guided generation and evaluation params
    parser.add_argument("--scale_coeff", type=float, default=0.0, help="Scale coefficient for noise rescaling.")
    parser.add_argument("--eval_num", type=int, default=16, help="Number of images for evaluation.")
    parser.add_argument("--image_classes", nargs='+', type=str, default=['cat', 'dog', 'horse', 'monkey', 'rabbit', 'butterfly', 'panda'], help="List of image classes.")
    parser.add_argument("--besteta", type=float, default=16.0, help="Eta value for guided generation (max in range).")
    parser.add_argument("--eta_min", type=float, default=8.0, help="Minimum eta value for random sampling in guided generation.")
    parser.add_argument("--cg_strength", type=float, default=75.0, help="Guidance strength for classifier guidance.")
    parser.add_argument("--sample_ckpt", type=str, default=None, help="Path to a checkpoint for sampling/initialization.")

    # System and I/O params
    parser.add_argument("--device", type=str, default='cuda', help="Device to use ('cuda' or 'cpu').")
    parser.add_argument("--seed", type=int, default=43, help="Random seed.")
    parser.add_argument("--init_latent_dir", type=str, default=f'{os.getcwd()}/data/cgimg/init', help="Directory for initial latents (aesthetic).")
    parser.add_argument("--output_dir_prefix", type=str, default=f'{os.getcwd()}/output/aes/', help="Prefix for DAgger iteration output directories (aesthetic).")

    # WandB params
    parser.add_argument("--wandb_project_name", type=str, default="GuidedImg-Aes-DAgger", help="WandB project name.")
    parser.add_argument("--wandb_run_name_prefix", type=str, default="DAgger_Aes_run", help="Prefix for WandB run name.")

    args = parser.parse_args()
    args.distributional = args.use_buckets
    return args

args = parse_aes_args()

# Update global config variables from args - This pattern is okay for scripts, but direct args access is cleaner.
# For this refactor, we'll keep it to ensure all original global usages are covered, then refine if needed.
# lr = args.lr
# batch_size = args.batch_size
# latent_dim = args.latent_dim
# num_inference_steps = args.num_inference_steps
# max_overall_iter = args.max_overall_iter
# epoches_per_iter = args.epoches_per_iter
# use_buckets = args.use_buckets
# num_bins = args.num_bins
# bin_min = args.bin_min
# bin_max = args.bin_max
# reinit_classifier = args.reinit_classifier
# device = args.device
# init_datapoint_num = args.init_datapoint_num
# generation_num = args.generation_num
# distributional = args.distributional
# scale_coeff = args.scale_coeff
# eval_num = args.eval_num
# image_classes = args.image_classes
# besteta = args.besteta
# eta_min = args.eta_min
# cg_strength = args.cg_strength
# sample_ckpt = args.sample_ckpt
setup_seed(args.seed)

config_from_args = vars(args)
logger.info(f"\nRunning DAgger for Aesthetics with parsed arguments:\n{config_from_args}")

convnet = SinusoidalTimeConvNet(args.latent_dim, num_classes=1, distributional=args.distributional, bin_min=args.bin_min, bin_max=args.bin_max, bin_num=args.num_bins).to(args.device)


# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss(reduction='none') if args.distributional else nn.MSELoss(reduction='none')
optimizer = torch.optim.Adam(convnet.parameters(), lr=args.lr)


sd_model.scheduler = DDIMScheduler.from_config(sd_model.scheduler.config)
sd_model.scheduler.set_timesteps(args.num_inference_steps) # DDIM Scheduler
timesteps = sd_model.scheduler.timesteps
scorer = AestheticScorerDiff(dtype=torch.float32).to(args.device)
resize = torchvision.transforms.Resize(224, antialias=False)
normalize = torchvision.transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                                std=[0.26862954, 0.26130258, 0.27577711])

# input do scale the noise
def rescale_std_inplace(std_src, noise_trg, scale_coeff=0.0):
    """
    Rescale `noise_trg` according to `std_src`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_trg = noise_trg.std(dim=list(range(1, noise_trg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes OOD issues)
    noise_coeff =  scale_coeff * std_src / std_trg + 1 - scale_coeff
    noise_trg.mul_(noise_coeff)
    return


# Dataset, sample latent, and the compressibility score of the decoded image
class AestheticLatentDataset(Dataset):
    ## Latent Structure: for init latent, we only store the latent at t=0, shape is [4, 64, 64]; for other latents, we store the latents from t = 0 to some t, t \le 50, the shape of latent is [t, 4, 64, 64]
    def __init__(self, latent_dir, on_the_air_dir=None, init_datapoint_num=args.init_datapoint_num):
        self.latent_dir = latent_dir
        self.reverse_timesteps = timesteps.flip(0)
        data_stages = os.listdir(on_the_air_dir) if on_the_air_dir is not None else []
        full_dirs = [os.path.join(on_the_air_dir, stage) for stage in data_stages] # each dir reperesent a stage
        full_dirs.append(latent_dir) # original init stage [path1, path2, ...], each path contains cat/ cat.npy, dog/ dog.npy, ...
        latent_classes = args.image_classes # only one class for now, but code support different class
        self.labels = [os.path.join(stage, latent_class +'_aes.npy') for stage in full_dirs for latent_class in latent_classes] # each label npy file
        self.reverse_labels_dic = {v: k for k, v in enumerate(self.labels)}  # for fast refer, given label npy file name return the index
        self.label_latents = [np.load(label) for label in self.labels] # load each label npy file, can be referred by index
        ## HERE TO CHANGE THE NUMBERS
        self.latent_files = [ os.path.join(stage, latent_class, f) 
                             for stage in full_dirs 
                             for latent_class in latent_classes 
                             for f in os.listdir(os.path.join(stage, latent_class))[:init_datapoint_num] if f.endswith('.npy')]
        
        print(f"Found {len(self.latent_files)} images in {latent_dir}")
        
    def __len__(self):
        return len(self.latent_files)

    def __getitem__(self, idx):
        latent_path = self.latent_files[idx]
        path, subid = os.path.split(latent_path)
        path = path + '_aes.npy'
        classid = self.reverse_labels_dic[path]
        subid = int(subid.split('.')[0])
        label = self.label_latents[classid][subid]
        latent = np.load(latent_path)
        if latent.ndim == 3:
            assert latent.shape[0] == 4, "Latent outside a batch should be of shape (4, 64, 64)"
            latent = torch.tensor(latent, dtype=torch.float32).unsqueeze(0)
            # Copy latent to shape [50, 4, 64, 64]
            latent = latent.repeat(50, 1, 1, 1)
            # image = decode_latents(latent[0])
            # compressibility = jpeg_compressibility(image.unsqueeze(0))
            random_noise = torch.randn_like(latent)
            latent = sd_model.scheduler.add_noise(latent, random_noise, self.reverse_timesteps)
            mask = torch.ones(len(latent), dtype=torch.float32)
        else:
            # print(latent.shape)
            latent = torch.tensor(latent, dtype=torch.float32)
            zero = torch.zeros(50 - latent.shape[0], *latent.shape[1:], dtype=torch.float32)
            mask = torch.cat([torch.ones(latent.shape[0], dtype=torch.float32), torch.zeros(50 - latent.shape[0], dtype=torch.float32)], dim=0)
            latent = torch.cat([latent, zero], dim=0)
        return latent, torch.tensor(label.repeat(len(latent)), dtype=torch.float32), mask


# Compute the gradient of the classifier with respect to the latent
@torch.enable_grad()
def compute_gradient(classifier, latent, timestep, eta):
    # print(latent.shape, timestep.shape, eta.shape)
    latent.requires_grad_(True)
    reward = classifier(latent, torch.tensor([timestep]).repeat(latent.shape[0]).to(args.device), eta)
    # print(reward)
    reward = reward.sum()
    classifier.zero_grad()
    reward.backward()
    grad = latent.grad.clone()
    classifier.zero_grad()
    # print(grad)
    return grad

@torch.no_grad()
def guided_generation(classifier, eta, guidance_scale=7.0, scale_coeff=args.scale_coeff, eval_num=args.eval_num, return_dis=False, random_seed=43, caption=None):
    setup_seed(random_seed)
    def callback(i, t, latents, classifier = classifier, eta = eta):
        prev_timestep = t - 20
        alpha_prod_t = sd_model.scheduler.alphas_cumprod[t]
        alpha_prod_t_prev = sd_model.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else torch.tensor(1.0)
        alpha_t = alpha_prod_t / alpha_prod_t_prev
        coeff = (1-alpha_t)/torch.sqrt(alpha_t)
        gradient = coeff * compute_gradient(classifier, latents, t, eta)
        # print(latents.mean(), (guidance_scale * gradient).mean())
        src_std = latents.std(dim=list(range(1, latents.ndim)), keepdim=True)
        latents.add_(guidance_scale * gradient)
        rescale_std_inplace(src_std, latents, scale_coeff)
        # print(latents.mean())
        return None
    image_list = []
    score_list = []
    for i in range(eval_num):
        label = random.choice(args.image_classes)
        if caption is not None:
            label = caption
        images = sd_model([label], callback=callback).images
        image_list.append(images[0])
        images = np.array(images[0])
        images = torch.from_numpy(images).permute(2, 0, 1).unsqueeze(0).to(args.device)
        images = images / 255.0
        images = resize(images)
        images = normalize(images)
        score_list.append(scorer(images)[0].item())
    if return_dis:
        return image_list, np.array(score_list)
    return image_list, np.array(score_list).mean()

@torch.no_grad()
def guided_generation_trajectory(classifier, eta, guidance_scale=7.0, label='cat'):
    traj = []
    transfer_time = random.randint(0, 49)
    def callback(i, t, latents, classifier = classifier, eta = eta, transfer_time = transfer_time):
        if i <= transfer_time:
            gradient = compute_gradient(classifier, latents, t, eta)
            src_std = latents.std(dim=list(range(1, latents.ndim)), keepdim=True)
            latents.add_(guidance_scale * gradient)
            rescale_std_inplace(src_std, latents, args.scale_coeff)
            if i == transfer_time:
                traj.append(latents.clone())
        else:
            traj.append(latents.clone())
        return None
    images = sd_model([label], callback=callback).images
    images = np.array(images[0])
    images = torch.from_numpy(images).permute(2, 0, 1).unsqueeze(0).to(args.device)
    images = images / 255.0
    images = resize(images).to(args.device)
    images = normalize(images)
    score = scorer(images)[0].item()
    traj = traj[::-1]
    traj = torch.stack(traj)
    traj = traj.squeeze(dim=1)

    return traj, score

if __name__ == "__main__":
    current_time_str = time.strftime("%Y%m%d-%H%M%S")
    wandb_run_name = f"{args.wandb_run_name_prefix}_{current_time_str}"
    wandb.init(project=args.wandb_project_name, name=wandb_run_name, config=config_from_args)

    run_specific_output_dir = os.path.join(args.output_dir_prefix, current_time_str)

    # ---------------------------
    # MAIN TRAINING LOOP
    # --------------------------
    os.makedirs(f'{run_specific_output_dir}/ckpts', exist_ok=True)
    os.makedirs(f'{run_specific_output_dir}/DAgger', exist_ok=True)
    for iteration in range(args.max_overall_iter):

        print(f"\n=== Overall Iteration {iteration} ===")

        # --- SAMPLE COLLECTION PHASE ---
        if iteration == 0 and args.sample_ckpt is None:
            print("Use existing dataset for the first iteration without classifier.")
        else:
            print("Prepare the dataset for the first iteration with classifier.")
            generated_path = f'{run_specific_output_dir}/DAgger/iteration_{iteration}'
            for latent_class in args.image_classes:
                scores = []
                latent_dir = os.path.join(generated_path, latent_class)
                os.makedirs(latent_dir, exist_ok=True)
                if iteration == 0:
                    convnet = torch.load(args.sample_ckpt).to(args.device)
                print(f"Generating samples for latent class {latent_class}...")
                for i in range(args.generation_num):
                    latent, score = guided_generation_trajectory(convnet, eta=random.uniform(args.eta_min, args.besteta), guidance_scale=args.cg_strength, label=latent_class)
                    latent = latent.cpu().numpy()
                    latent_path = os.path.join(latent_dir, f'{i}.npy')
                    np.save(latent_path, latent)
                    scores.append(score)
                np.save(os.path.join(generated_path, f'{latent_class}_aes.npy'), np.array(scores))


            if args.reinit_classifier:
                print('reinit model')
                convnet = SinusoidalTimeConvNet(args.latent_dim, num_classes=1, distributional=args.distributional, bin_min=args.bin_min, bin_max=args.bin_max, bin_num=args.num_bins).to(args.device)
                optimizer = torch.optim.Adam(convnet.parameters(), lr=args.lr)

        train_dataset = AestheticLatentDataset(args.init_latent_dir, f'{run_specific_output_dir}/DAgger')
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True)

        # --- CLASSIFIER TRAINING PHASE ---
        print("Training the classifier...")
        for epoch in range(args.epoches_per_iter):
            convnet.train()
            epoch_loss = 0.0
            for i, (inputs, targets, masks) in enumerate(tqdm(train_loader, desc="Training Progress")):
                inputs, targets, masks = inputs.to(args.device), targets.to(args.device), masks.to(args.device)
                inputs = inputs.view(-1, *inputs.shape[2:])
                targets = targets.view(-1)
                masks = masks.view(-1).unsqueeze(1)
                # print(inputs.shape, targets.shape) should be [B*50, 4, 64, 64], [B*50]
                # Forward pass
                # print(inputs.shape, timesteps.repeat(batch_size).shape)
                outputs = convnet(inputs, timesteps.repeat(args.batch_size).to(args.device))# [batch_size*50, bin_num]

                # Compute the loss
                if args.distributional:
                    # if CrossEntropyLoss, we need to convert the targets to long
                    bins = torch.linspace(args.bin_min, args.bin_max, args.num_bins+1).to(args.device)
                    targets = torch.bucketize(targets, bins) - 1
                    targets = targets.clamp(min=0, max=args.num_bins - 1)
                    # print(outputs.shape, targets.long())
                    loss = (criterion(outputs, targets.long()) * masks.view(-1)* masks).sum() / masks.sum() / convnet.bin_num
                else:
                    loss = (criterion(outputs.view(-1), targets) * masks.view(-1)* masks).sum() / masks.sum()
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                wandb.log({"iter_loss": loss.item(), "epoch": epoch, "iter": i})
                logger.info(f"iter_loss: {loss.item()}")

                epoch_loss += loss.item()
                # if i > 10:
                #     break
            wandb.log({"epoch_loss": epoch_loss/(i+1)})
            logger.info(f"epoch_loss: {epoch_loss/(i+1):.4f}")
            
            torch.save(convnet, f'{run_specific_output_dir}/ckpts/reward_predictor_overalliter_{iteration}_epoch_{epoch}.pth')

            # --- GUIDED EVALUATION PHASE ---
            print("Guided evaluation with multi case...")
            convnet.eval()
            if args.use_buckets:
                max_reward = -np.inf
                for eta in torch.linspace(start=0, end=20, steps=11):
                    eta_val = eta.item()
                    print(f"Evaluating for eta = {eta_val:.4f}")
                    images, mean_reward = guided_generation(convnet, eta_val, guidance_scale = args.cg_strength)
                    image_to_show = paste_square(images, grid_size=4)
                    mean_reward = torch.tensor(mean_reward).to(args.device)
                    wandb.log({"epoch": epoch,
                            "eta": eta_val,
                            "guided_images": wandb.Image(image_to_show, caption=f"Guided images for eta = {eta_val:.4f}"),
                            "mean_reward": mean_reward})
                    print(f"Epoch {epoch}: Mean reward for eta = {eta_val:.4f} is {mean_reward:.4f}")
                    if mean_reward > max_reward:
                        max_reward = mean_reward
                        max_eta = eta_val
                print(f"Epoch {epoch}: Max reward = {max_reward:.4f} for eta = {max_eta:.4f}")
            else:
                print('Error, only support buckets for now')
    logger.info("DAgger aesthetic training process completed.")
    wandb.finish()
