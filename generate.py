import torch 
import torch.nn as nn
from argparse import ArgumentParser
from torchvision import transforms as T
import time

from model import CoAtNet
from diffusion_helper import generate, brain2image
from diffusion_helper import prepare_diffuser, prepare_text_embedding
from utils import read_config, load_spectos

parser = ArgumentParser(description="Braincoder Eval Generation Test")
parser.add_argument("--participant", type=str)
# parser.add_argument("--image`")
parser.add_argument("--checkpoint_dir", type=str)
parser.add_argument("--cfg", type=str, default=None)
parser.add_argument("--out", type=str)
parser.add_argument("--im_dir", type=str)

args = parser.parse_args()
checkpoint_dir = args.checkpoint_dir
cfg_path = args.cfg
out_dir = args.out
participant = args.participant
im_base_dir = args.im_dir

device = 'cuda' if torch.cuda.is_available() else "cpu"
cfg = read_config(cfg_path)
output_scale = cfg["exp"]["output_scale"]

transform = T.Compose([
    T.ToTensor()
])

model = CoAtNet.from_trained(cfg["model"], checkpoint_dir)
vae, unet, scheduler = prepare_diffuser(device)

for img_id in range(20):
    for con in ["start", 'middle', "end"]:
        start = time.time()

        paths = [f"{im_base_dir}/{participant}_{img_id+1}_{con}_c_{channel}" for channel in range(14)]
        spectos = load_spectos(paths, transform, device=device)

        embedding = model(spectos) / output_scale
        
        generated_image = brain2image(embedding, vae=vae, unet=unet, scheduler=scheduler, device=device)[0]

        generated_image.save(f"{out_dir}/{participant}_{img_id+1}_{con}.png")

        print(f"Generate Done in {time.time() - start}")