from diffusers import StableDiffusionPipeline

sd_model = StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", safety_checker=None, requires_safety_checker=False)#, local_files_only=True)
sd_model.to('cuda') 
sd_model.vae.requires_grad_(False)
sd_model.text_encoder.requires_grad_(False)
sd_model.unet.requires_grad_(False)
sd_model.safety_checker = None

# sd_model.vae = sd_model.vae.to('cuda') # only transfer vae to GPU for just encode and decode images
sd_model.vae.eval()
sd_model.text_encoder.eval()
sd_model.unet.eval()

def encode(im):
    if im.device != 'cuda':
        im = im.to('cuda')
    latent = sd_model.vae.encode(im).latent_dist.sample()
    latent = sd_model.vae.config.scaling_factor * latent
    return latent.cpu()

# input torch latents: [B, C, H, W], output image is also torch tensor: [B, C, H, W]
def decode_latents(latents):
    if latents.device != 'cuda':
        latents = latents.to('cuda')
    latents = 1 / sd_model.vae.config.scaling_factor * latents
    image = sd_model.vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
    # image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    return image.cpu() # [B, C, H, W] in 'cpu'
