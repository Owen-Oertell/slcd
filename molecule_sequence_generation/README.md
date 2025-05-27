# Molecule Sequence Generation

This is the submodule for the molecule sequence generation tasks.

### Installation

First, install the dependencies in a fresh conda enviroment.

```bash
conda create -n slcd python=3.10
conda activate slcd
pip install uv
export UV_TORCH_BACKEND=auto # for auto selecting the torch backend
uv pip install torch==2.5.1 # you can change this for the version of cuda you have.
uv pip install -r requirements.txt
```

Then, to download the base models, run:
```bash
python load_models.py
```

This will download the base models to the `artifacts` directory. For both sequence tasks, generation is unconditional so we don't need a dataset.

To train the classifier/value function for the DNA task, run:
```bash
python train.py train.learning_rate=0.0001 \
 train.train_iterations=200 \
 train.classifier_batch_size=5 \
 train.grad_accumulation_steps=4 \
 model.bucketing=True \
 train.reset_dataset=False \
 train.reinit_classifier=False \
 model.collapse_to_mean=True \
 train.task="dna"
```

To train the classifier/value function for the RNA task, run:
```bash
python train.py train.learning_rate=0.0001 \
 train.train_iterations=200 \
 train.classifier_batch_size=5 \
 train.grad_accumulation_steps=4 \
 model.bucketing=True \
 train.reset_dataset=False \
 train.reinit_classifier=False \
 model.collapse_to_mean=True \
 train.task="rna"
```

You can use the `plotting.py` script to plot the training curves.

To generate samples, you can run the decode scripts. `decode_classifier.py` will allow you use the SLCD method to generate samples. On the other hand, `decode.py` will use SVDD-MC, but using the newly trained classifier. In our experiments, we found that SLCD outperforms SVDD-MC with the same model.

### Hyperparameters

The following table lists the hyperparameters found in `conf/config.yaml` and their descriptions:

| Hyperparameter             | Meaning                                                                         |
|----------------------------|---------------------------------------------------------------------------------|
| `seed`                     | Random seed for reproducibility.                                                |
| `wandb`                    | Boolean flag to enable/disable Weights & Biases logging.                        |
| `name`                     | Name of the experiment run.                                                     |
| `override_name`            | Boolean flag to override the default naming convention for the run.               |
| `exit_after_first_epoch`   | Boolean flag to stop training after the first epoch (likely for debugging).     |
| `rank`                     | Process rank for distributed training.                                          |
| `world_size`               | Total number of processes for distributed training.                             |
| `model.n_tasks`            | Number of output tasks or targets for the model.                                |
| `model.bucketing`          | Boolean flag to enable/disable distributional prediction.                       |
| `model.collapse_to_mean`   | Boolean flag to collapse distributional output to a single mean.                |
| `train.learning_rate`      | The learning rate for the optimizer.                                            |
| `train.weight_decay`       | The weight decay (L2 penalty) for the optimizer.                                |
| `train.betas`                | Coefficients for computing running averages of gradient and its square (Adam optimizer). |
| `train.num_epochs`           | The total number of training epochs.                                            |
| `train.lr_decay`             | Boolean flag to enable/disable learning rate decay.                             |
| `train.inference_batch_size` | Batch size used during inference.                                               |
| `train.classifier_batch_size`| Batch size used for training the classifier.                                    |
| `train.num_classifier_epochs`| Number of epochs for training the classifier.                                   |
| `train.task`                 | Specifies the task type: "rna" or "dna".                         |
| `train.val_batch_num`        | Number of validation batches to use during evaluation.                          |
| `train.method`               | Method for training, either "svdd" or "gradient".                               |
| `train.svdd_sample_M`        | Number of samples M for SVDD method.                                           |
| `train.grad_accumulation_steps` | Number of steps to accumulate gradients before an optimizer update.             |
| `train.grad_norm_clip`       | Maximum norm for gradient clipping (-1 to disable).                             |
| `train.log_interval`         | Interval (in steps/iterations) for logging training progress.                   |
| `train.save_checkpoint`      | Boolean flag to enable/disable saving model checkpoints.                        |
| `train.guidance_scale`       | Scale factor for classifier guidance during generation.                         |
| `train.train_iterations`     | Number of iterations for generating training samples.                           |
| `train.initial_train_iterations` | Initial number of iterations for generating training samples.                   |
| `train.classifier_test_iterations` | Number of iterations for generating test samples for the classifier.          |
| `train.eval_interval`        | Interval (in steps/iterations) for performing evaluation.                       |
| `train.compute_rewards_interval` | Interval (in steps/iterations) for computing rewards.                         |
| `train.reinit_classifier`    | Boolean flag to reinitialize the classifier before training.                    |
| `train.reset_dataset`        | Boolean flag to reset the dataset.                                              |
| `train.cdq`                  | Boolean flag for Continuous Diffusion Quantization.                             |

### Acknowledgement  

Our codebase is directly built on top of [SVDD](https://github.com/masa-ue/SVDD)  

