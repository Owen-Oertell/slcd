from diffusers import StableDiffusionPipeline
from diffusers import DDIMScheduler
from tqdm import tqdm
import os
import torch
import numpy as np
import argparse

from vae import decode_latents
from compressibility_scorer import jpeg_compressibility

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images with Stable Diffusion.")
    parser.add_argument("--output_dir", type=str, default="cgimg/init", help="Directory to save generated images and latents.")
    parser.add_argument("--prompt", type=str, default="cat", help="Prompt to generate images.")
    args = parser.parse_args()
    prompts = ['cat', 'dog', 'horse', 'monkey', 'rabbit', 'butterfly', 'panda']
    setup_seed(42)

    device = 'cuda'
    pipeline = StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5")#, local_files_only=True)
    pipeline.to(device)
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline.scheduler.set_timesteps(50, device=device)
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.unet.requires_grad_(False)

    pipeline.vae.eval()
    pipeline.text_encoder.eval()
    pipeline.unet.eval()
    
    for p in prompts:
        output_path = os.path.join(args.output_dir, p)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        for i in tqdm(range(10000), desc="Generating Images"):
            latent = pipeline(p, output_type='latent').images[0]
            latent = np.array(latent.cpu())
            np.save(os.path.join(output_path, f"{i}.npy"), latent)
        