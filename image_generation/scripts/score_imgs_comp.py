import torch
import numpy as np
from vae import decode_latents
from tqdm import tqdm
from compressibility_scorer import jpeg_compressibility
import argparse
import os

device = 'cuda'
comp_scorer = jpeg_compressibility


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score generated images based on aesthetic quality.")
    parser.add_argument("--base_path", type=str, default="cgimg/init/", 
                        help="Base directory where latent files (grouped by prompt) are stored.")
    parser.add_argument("--prompts", nargs='+', type=str, 
                        default=['cat', 'dog', 'horse', 'monkey', 'rabbit', 'butterfly', 'panda'],
                        help="List of prompt strings for which to score images.")
    parser.add_argument("--num_images", type=int, default=10000, 
                        help="Number of images to score per prompt.")
    parser.add_argument("--output_suffix", type=str, default=".npy", 
                        help="Suffix for the output numpy file containing scores (e.g., '_aes.npy').")
    
    args = parser.parse_args()

    for prompt in args.prompts:
        scores = []
        prompt_latent_path = os.path.join(args.base_path, prompt)
        
        print(f"Scoring images for prompt: {prompt} from {prompt_latent_path}")

        if not os.path.isdir(prompt_latent_path):
            print(f"Warning: Directory not found {prompt_latent_path}, skipping prompt: {prompt}")
            continue

        for i in tqdm(range(args.num_images), desc=f"Scoring Images for {prompt}"):
            latent_file_path = os.path.join(prompt_latent_path, str(i) + '.npy')
            
            if not os.path.exists(latent_file_path):
                continue
            
            try:
                latent = np.load(latent_file_path)
                latent = torch.tensor(latent, dtype=torch.float32).unsqueeze(0)
                image = decode_latents(latent)
                scores.append(comp_scorer(image))
            except Exception as e:
                print(f"Error processing file {latent_file_path}: {e}")
                continue

        if not scores:
            print(f"No scores collected for prompt: {prompt}. Output file will not be saved.")
            continue

        output_file_name = prompt + args.output_suffix
        output_save_path = os.path.join(args.base_path, output_file_name)
        
        np.save(output_save_path, np.array(scores))
        print(f"Saved scores for {prompt} to {output_save_path}")