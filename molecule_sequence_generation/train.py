# TRAINING SCRIPT FOR RNA/DNA Diffusion Models

import os
import hydra
from omegaconf import DictConfig, OmegaConf
from Enformer import (
    BaseModel,
    ConvHead,
    EnformerTrunk,
)
from trainer import Trainer
import torch
import numpy as np
import random

# hide warnings
import warnings
warnings.filterwarnings("ignore")

@hydra.main(config_path="conf", config_name="config.yaml", version_base=None)
def main(cfg: DictConfig):
    
    # set rank
    cfg.rank = int(os.environ.get("RANK", 0))
    cfg.world_size = int(os.environ.get("WORLD_SIZE", 1))

    if cfg.rank == 0:
        print("-- Starting Run with Config --")
        config_yaml = OmegaConf.to_yaml(cfg, resolve=True)
        print(config_yaml)
        print("--------------------------------")

    # set seed
    seed = cfg.seed + cfg.rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Initalize the Model (enformer)
    common_trunk = EnformerTrunk(
        n_conv=7,
        channels=1536,
        n_transformers=11,
        n_heads=8,
        key_len=64,
        attn_dropout=0.00,
        pos_dropout=0.00,
        ff_dropout=0.0,
        crop_len=0,
    )
    
    reg_head = ConvHead(
        n_tasks=cfg.model.n_tasks,
        in_channels=2 * 1536,
        act_func=None,
        pool_func="avg",
        distributional=cfg.model.bucketing,
    )

    # common_trunk = classifier.NewTrunk()
    # reg_head = classifier.NewHead()


    model = BaseModel(
        embedding=common_trunk,
        head=reg_head,
        cdq=cfg.train.cdq,
        batch_size=cfg.train.inference_batch_size,
        val_batch_num=cfg.train.val_batch_num,
        task=cfg.train.task,
        n_tasks=cfg.model.n_tasks,
        distributional=cfg.model.bucketing,
    )

    if cfg.wandb and cfg.rank == 0:
        import wandb
        config_dict = OmegaConf.to_container(cfg, resolve=True)

        if cfg.override_name:
            name = cfg.name
        else:
            name = f"{cfg.name}_it{cfg.train.train_iterations}_tiz{cfg.train.train_iterations}_lr{cfg.train.learning_rate}_wd{cfg.train.weight_decay}_lr_decay{cfg.train.lr_decay}"
        wandb.init(project="PROJECT_NAME", config=config_dict, name=name, entity="owen-oertell")

    # print number of parameters
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    trainer = Trainer(model, cfg)
    trainer.setup()
    trainer.fit()

if __name__ == "__main__":
    main()