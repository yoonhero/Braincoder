import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import wandb
import h5py

import tqdm
import time
import os
from argparse import ArgumentParser
from pathlib import Path

from model import LinearModel, CoAtNet
from dataloader import create_dataloader, COCOCOCOCOCCOCOOCOCOCOCCOCOCOCODatset
from diffusion_helper import generate, prepare_diffuser, prepare_text_embedding
from utils import read_config, load_spectos
# from train import loss_term

# Get SYS arguments of config file loading.
parser = ArgumentParser(description="braincoder training pipeline")

deafult_model_name = "linear"
parser.add_argument("--cfg", default="./config/exp_config_yaml", type=str)
# parser.add_argument("--transfer", action="store_true")

args = parser.parse_args()
cfg_file_path = args.cfg

# ================= HYPER PARAMETERS ================
models = {
    "linear": LinearModel,
    "coatnet": CoAtNet
}
device = "cuda" if torch.cuda.is_available() else "cpu"

cfg = read_config(cfg_file_path)
exp_cfg = cfg["exp"]
model_cfg = cfg["model"]

model_name = exp_cfg["model_name"]

participant = exp_cfg["participant"]

learning_rate = exp_cfg["learning_rate"] 
batch_size = exp_cfg["batch_size"]
epochs = exp_cfg["epochs"]

try:
    grad_accum = exp_cfg["grad_accum"]
except: 
    grad_accum = 1

b1, b2 = exp_cfg["betas"]
alpha = exp_cfg["alpha"]

cache_dir = exp_cfg["cache_dir"]
pretrain_dir = exp_cfg["pretrain_dir"]
checkpoint_dir = exp_cfg["checkpoint_dir"]
image_dir = exp_cfg["image_dir"]

metrics = exp_cfg["metrics"]
optimizer_type = exp_cfg["optimizer"]
weight_decay = exp_cfg["weight_decay"]
grad_clip = exp_cfg["grad_clip"]

exp_name = exp_cfg["exp_name"]
Path(checkpoint_dir).mkdir(exist_ok=True)
(Path(checkpoint_dir)/str(exp_name)).mkdir(exist_ok=True)

seed = exp_cfg["seed"]

valid_term = 1
save_term = exp_cfg["save_term"]

output_scale = exp_cfg["output_scale"]

print(cfg)

# =================== DATASET LOADING =====================
class FineTuneDataset(Dataset):
    def __init__(self, participant, image_dir, scale, device, cache_dir=None):
        self.device = device
        self.image_dir = image_dir
        self.scale = scale

        self.transforms = T.Compose([
            T.Resize((240, 320)),
            T.ToTensor()
        ])

        start_time_sec = time.time()
        
        dataset = []
        for con in ["start", "middle", "end"]:
            for im_id in range(1, 21):
                _spec = [f"{self.image_dir}/{participant}_{im_id}_{con}_c_{c}.png" for c in range(14)]
                if not os.path.exists(_spec[0]): continue
                dataset.append(im_id, _spec, "")

        # id,src,caption,start,end,width,height, spectogram
        self.dataset = dataset

        self.load_cache(cache_dir)

        print(f"Finishing Loading Dataset in {time.time() - start_time_sec}s")
    
    def __getitem__(self, index):
        im_id, specto, caption = self.dataset[index]

        x = load_spectos(specto, self.transforms, self.device)

        y = self.get_emb_from_cache(im_id)
        y = torch.from_numpy(y[:]).squeeze(0).to(self.device)

        # clip has too large value, so I decide to normalize the vector for efficient predicting.
        y *= self.scale

        return x, y, im_id
    
    def load_cache(self, cache_dir):
        f = h5py.File(cache_dir, "r")
        self.cache_dataset = f["data"]

    def get_emb_from_cache(self, id):
        return self.cache_dataset[str(id)]
        
    def __len__(self):
        return len(self.dataset)


train_dataset = FineTuneDataset(participant, image_dir, output_scale, devcie=device, cache_dir=cache_dir)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

to_samples = []
to_samples_keys = []


# ================== Load Main Model =================
model = models[model_name].from_pretrained(model_cfg, pretrain_dir, True).to(device)
total_parameters = model.num_parameters()


# ================== WANDB LOGGER =================
run = wandb.init(
  project="braincoder",
  config={**model_cfg, **exp_cfg, "param": total_parameters}
)


# ================== Training Loop =================
def loss_term(y, y_hat):
    mse_loss = ((y_hat-y)**2).mean()

    if "kl" in metrics:
        epsilon = 1e-10
        q_distribution_clamped = torch.clamp(y_hat, min=epsilon)
        p_distribution_clamped = torch.clamp(y, min=epsilon)
        kl_loss = F.kl_div(q_distribution_clamped.log(), p_distribution_clamped, reduction='batchmean')
        loss = mse_loss + (1-alpha) * kl_loss
    elif "cross_en" in metrics:
        cross_loss = F.cross_entropy(y_hat, y)
        loss = mse_loss + (1-alpha)*cross_loss
    elif "cos" in metrics:
        cos_loss = 1 - torch.cosine_similarity(y_hat, y, dim=-1).mean()
        loss = alpha*mse_loss + (1-alpha)*cos_loss

    return loss

def train_one_epoch(model, loader, optimizer):
    _loss = []

    for step, batch in enumerate(tqdm.tqdm(loader)):
        x, y, im_key = batch
        yhat = model(x)

        loss = loss_term(y, yhat)
        loss /= grad_accum

        loss.backward()
        _loss.append(loss.item())

        if (step+1)%grad_accum==0 or step+1==len(loader):
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip, norm_type=2)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
                

    mean_loss = sum(_loss) / len(_loss)
    return mean_loss


def finetune():
    if optimizer_type == "Adam":
        optimizer = torch.optim.Adam(model.get_parameters(weight_decay), lr=learning_rate, weight_decay=weight_decay, betas=(b1, b2))
    elif optimizer_type == "AdamW":
        optimizer = torch.optim.AdamW(model.get_parameters(weight_decay), lr=learning_rate, weight_decay=weight_decay, betas=(b1, b2))
    elif optimizer_type == "SGD":
        optimizer = torch.optim.SGD(model.get_parameters(weight_decay), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)

    for epoch in range(epochs):
        train_loss = train_one_epoch(model=model, loader=train_loader, optimizer=optimizer)
        run.log({"train/loss": train_loss})
        
        # SAVE the checkpoint
        if (epoch+1)%save_term == 0:
            torch.save(model.state_dict(), f"{checkpoint_dir}/{exp_name}/epoch-{epoch}.pt")


finetune()