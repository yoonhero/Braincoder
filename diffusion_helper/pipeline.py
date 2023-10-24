import torch
import torch.nn as nn
from PIL import Image
from functools import lru_cache
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers import LMSDiscreteScheduler

from diffusion_helper import prepare_diffuser, prepare_text_embedding, text2emb

def generate(embedding, vae: AutoencoderKL, unet: UNet2DConditionModel, scheduler: LMSDiscreteScheduler, device, num_inference_steps=100, guidance_scale=7.5):
    width, height = 512, 512
    generator = torch.manual_seed(0)    # Seed generator to create the inital latent noise
    scheduler.set_timesteps(num_inference_steps)

    # embedding first dimension could be the double of the prompt list
    batch_size = embedding.shape[0] // 2

    latents = torch.randn(
        (batch_size, unet.in_channels, height // 8, width // 8),
        generator=generator,
    )
    latents = latents.to(device)

    B, seq_len, _ = embedding.shape
    encoder_attention_mask_no_masking = torch.ones((B, seq_len))

    for t in tqdm(scheduler.timesteps):
        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        # for classifier-free guidance process we need to latents this can get better performance.
        latent_model_input = torch.cat([latents] * 2)

        latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

        # predict the noise residual
        with torch.no_grad():
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=embedding, encoder_attention_mask=encoder_attention_mask_no_masking).sample

        # classifier-free guidance implementation
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    # scale and decode the image latents with vae
    latents = 1 / 0.18215 * latents
    with torch.no_grad():
        image = vae.decode(latents).sample

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def text2img(prompt, device="cuda"): 
    vae, unet, scheduler = prepare_diffuser(device)
    tokenizer, text_encoder = prepare_text_embedding(device)

    batch_size = len(prompt)

    text_tokened, text_embeddings = text2emb(prompt, tokenizer=tokenizer, text_encoder=text_encoder, device=device)

    max_length = text_tokened.input_ids.shape[-1]
    uncond_input = tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]

    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    images = generate(text_embeddings, vae=vae, unet=unet, scheduler=scheduler, device=device)

    return images



if __name__ == "__main__":
    prompt = ["Three teddy bears sit on a fake sled in fake snow"]
    images = text2img(prompt)

    plt.imshow(images[0])