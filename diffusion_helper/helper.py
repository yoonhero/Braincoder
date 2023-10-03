import time
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from diffusers import LMSDiscreteScheduler
from functools import lru_cache

@lru_cache()
def prepare_diffuser(device="cuda"):
    start = time.time()
    # 1. Load the autoencoder model which will be used to decode the latents into image space. 
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")

    # 2. The UNet model for generating the latents.
    unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")

    # diffusion process scheduler
    scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)

    vae.to(device)
    unet.to(device) 

    print(f"Load Diffusion Model Finished in {time.time() - start}")

    return vae, unet, scheduler


def prepare_text_embedding(device):
    # 2. Load the tokenizer and text encoder to tokenize and encode the text. 
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

    text_encoder.to(device)

    return tokenizer, text_encoder


def text2emb(text, tokenizer:CLIPTokenizer, text_encoder:CLIPTextModel, device):
    # compute the text embedding vector for the conditional generation
    text_tokened = tokenizer(text, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    text_embeddings = text_encoder(text_tokened.input_ids.to(device))[0]

    return text_tokened, text_embeddings



