import torch 
import torch.nn as nn
from argparse import ArgumentParser

from model import CoAtNet
from diffusion_helper import generate
from diffusion_helper import prepare_diffuser, prepare_text_embedding
from utils import read_config

parser = ArgumentParser(description="Braincoder Generation Test")
parser.add_argument("--emb_path", type=str)
# parser.add_argument("--image`")
parser.add_argument("--checkpoint_dir", type=str)
parser.add_argument("--cfg", type=str, default=None)
parser.add_argument("--out", type=str)

args = parser.parse_args()
embedding_path = args.emb_path
checkpoint_dir = args.checkpoint_dir
cfg_path = args.cfg
out_dir = args.out

if cfg_path != None:
    cfg = read_config(cfg_path)

def load_emb(emb_path, device):
    return torch.load(emb_path).to(device)

device = 'cuda' if torch.cuda.is_available() else "cpu"

##### PREPARE ######

if checkpoint_dir != None:
    #### TODODODODOODODOODODOo
    network = CoAtNet.from_trained(cfg, checkpoint_dir)
else:
    embedding = load_emb(embedding_path, device=device)
    embedding = embedding.unsqueeze(0)

vae, unet, scheduler = prepare_diffuser(device)

generated_images = generate(embedding, vae=vae, unet=unet, scheduler=scheduler, device=device)

for i, image in enumerate(generated_images):
    image.save(f"{out_dir}/{i}.png")