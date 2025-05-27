import argparse
import numpy as np
import torch
import wandb
import random

from Enformer import BaseModel, BaseModelMultiSep, ConvHead, EnformerTrunk, TimedEnformerTrunk

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def run(args):
    set_seed(args.seed)
    wandb.init(
        project="RNA-optimization",
        job_type='FA',
        name='decode_classifier' if not args.run_name else args.run_name,
        config=vars(args)
    )

    print("Loading model")
    if args.model == 'enformer':
        common_trunk = EnformerTrunk(n_conv=7, channels=1536, n_transformers=11, n_heads=8, key_len=64,
                                     attn_dropout=0.05, pos_dropout=0.01, ff_dropout=0.4, crop_len=0)
        reg_head = ConvHead(n_tasks=1, in_channels=2 * 1536, act_func=None, pool_func='avg', distributional=args.distributional)
        model = BaseModel(embedding=common_trunk, head=reg_head, cdq=args.cdq, batch_size=args.batch_size, val_batch_num=1, task=args.task, n_tasks=args.n_task, saluki_body=args.saluki_body, distributional = args.distributional)
    elif args.model == 'multienformer':
        common_trunk = EnformerTrunk(n_conv=7, channels=1536, n_transformers=11, n_heads=8, key_len=64,
                                     attn_dropout=0.05, pos_dropout=0.01, ff_dropout=0.4, crop_len=0)
        reg_head = ConvHead(n_tasks=1, in_channels=2 * 1536, act_func=None, pool_func='avg')
        model = BaseModelMultiSep(embedding=common_trunk, head=reg_head, cdq=args.cdq, batch_size=args.batch_size, val_batch_num=args.val_batch_num, distributional = args.distributional)
    elif args.model == 'timedenformer':
        common_trunk = TimedEnformerTrunk(n_conv=7, channels=1536, n_transformers=11, n_heads=8, key_len=64,
                                     attn_dropout=0.05, pos_dropout=0.01, ff_dropout=0.4, crop_len=0)
        reg_head = ConvHead(n_tasks=1, in_channels=2 * 1536, act_func=None, pool_func='avg')
        model = BaseModel(embedding=common_trunk, head=reg_head, cdq=args.cdq, batch_size=args.batch_size,
                          val_batch_num=args.val_batch_num, timed=True, distributional = args.distributional)
    else:
        raise NotImplementedError(f"Model {args.model} not implemented")

    if args.pre_model_path:
        print(f"Loading pretrained model: {args.pre_model_path}")
        model.load_state_dict(torch.load(args.pre_model_path, map_location='cpu')['model_state_dict'], strict=True)
    
    if args.load_checkpoint_path:
        print(f"Loading stored model: {args.load_checkpoint_path}")
        checkpoint = torch.load(args.load_checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    
    print(f'Total params: {sum(p.numel() for p in model.parameters())}')

    model.cuda()
    if args.task in ("rna_saluki", "rna"):
        model.eval()
        if hasattr(model.embedding, 'gru_tower') and hasattr(model.embedding.gru_tower, 'gru'):
             model.embedding.gru_tower.gru.train() # Specific handling for GRU layers
    else:
        model.eval()

    reward_model_preds = model.controlled_decode_classfier(gen_batch_num=args.val_batch_num, guidance_scale=args.guidance_scale, collapse_to_mean=args.collapse_to_mean, eta=args.eta)
    
    top_50_percentile = np.percentile(reward_model_preds.cpu().numpy(), 50)
    reward_model_preds_top_50 = reward_model_preds[reward_model_preds > top_50_percentile]
    print(f"Mean of top 50% reward model preds: {reward_model_preds_top_50.mean()}")

    top_10_percentile = np.percentile(reward_model_preds.cpu().numpy(), 90) # Corrected to use original preds for 90th percentile
    reward_model_preds_top_10 = reward_model_preds[reward_model_preds > top_10_percentile]
    print(f"Mean of top 10% reward model preds: {reward_model_preds_top_10.mean()}")

    hepg2_values_ours = reward_model_preds.cpu().numpy()

    np.savez(f"./log/{args.task}-{args.reward_name}-classfier-p2.npz", decoding=hepg2_values_ours, baseline=hepg2_values_ours)
    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run model decoding with classifier guidance.')

    # Essential arguments
    parser.add_argument('--model', type=str, default='enformer', choices=['enformer', 'multienformer', 'timedenformer'], help="Model architecture to use.")
    parser.add_argument('--task', type=str, default="rna_saluki", help="Task identifier.")
    parser.add_argument('--guidance_scale', type=float, required=True, help="Guidance scale for classifier-free guidance.")

    # Paths
    parser.add_argument('--load_checkpoint_path', type=str, default=None, help="Path to load a training checkpoint.")
    parser.add_argument('--pre_model_path', type=str, default=None, help="Path to a pretrained model.")
    
    # Batch and data related
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for decoding.")
    parser.add_argument('--val_batch_num', type=int, default=1, help="Number of validation batches to generate.")
    
    # Model specific parameters (defaults are for enformer)
    parser.add_argument('--n_task', type=int, default=1, help="Number of tasks for the model head.")
    parser.add_argument('--saluki_body', type=int, default=0, help="Saluki body identifier (if applicable).")
    parser.add_argument('--distributional', action='store_true', help='Enable distributional output for the model.')
    parser.add_argument('--cdq', action='store_true', help='Enable CD-Q.')
    parser.add_argument('--collapse_to_mean', action='store_true', help='Collapse distributional prediction to mean.')
    parser.add_argument('--eta', type=float, default=0, help="Eta parameter for certain decoding strategies.")

    # Miscellaneous
    parser.add_argument('--run_name', type=str, default=None, help="Custom name for the W&B run.")
    parser.add_argument('--seed', type=int, default=44, help="Random seed for reproducibility.")
    parser.add_argument('--reward_name', type=str, default='HepG2', help="Name for the reward/metric being evaluated (used in output file name).")

    args = parser.parse_args()
    run(args)
