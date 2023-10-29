import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import wandb
import tqdm

from argparse import ArgumentParser
from pathlib import Path

from model import LinearModel, CoAtNet
from dataloader import create_dataloader
from diffusion_helper import generate, prepare_diffuser, prepare_text_embedding
from utils import read_config

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

learning_rate = exp_cfg["learning_rate"] 
batch_size = exp_cfg["batch_size"]
epochs = exp_cfg["epochs"]

try:
    grad_accum = exp_cfg["grad_accum"]
except: 
    grad_accum = 1

b1, b2 = exp_cfg["betas"]
alpha = exp_cfg["alpha"]

just_one = exp_cfg["just_one_pre_run"]

cache_dir = exp_cfg["cache_dir"]
checkpoint_dir = exp_cfg["checkpoint_dir"]
image_dir = exp_cfg["image_dir"]

num_to_samples = exp_cfg["num_to_samples"]
how_many_to_save = exp_cfg["how_many_to_save"]

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
train_loader, eval_loader = create_dataloader(batch_size=batch_size, cache_dir=cache_dir, scale=output_scale, image_dir=image_dir, device=device, seed=seed)

to_samples = []
to_samples_keys = []


# ================== Load Main Model =================
model = models[model_name].from_cfg(model_cfg).to(device)
total_parameters = model.num_parameters()


# ================== WANDB LOGGER =================
run = wandb.init(
  project="braincoder",
  config={**model_cfg, **exp_cfg, "param": total_parameters}
)


# ================== Training Loop =================

# Define the LOSS based on the CONFIG.
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

        if just_one or (step+1)%grad_accum==0 or step+1==len(loader):
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip, norm_type=2)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
                

    mean_loss = sum(_loss) / len(_loss)
    return mean_loss


@torch.no_grad()
def evaluation(epoch):
    _loss = []
    _saved = 0

    for batch in eval_loader:
        x, y, im_keys = batch
        yhat = model(x)
        loss = loss_term(y, yhat)
        _loss.append(loss.cpu().detach().item())

        # Saving the prediction for concise model test process.
        for i in range(how_many_to_save):
            if i >= yhat.shape[0] or _saved >= how_many_to_save:
                continue
            pred = yhat[i].cpu().detach()
            im_key = im_keys[i].item()
            filename = f"{checkpoint_dir}/{exp_name}/sample-{epoch}-{im_key}.pt"
            torch.save(pred, filename)
            _saved+=1

    mean_loss = sum(_loss) / len(_loss)
    return mean_loss


# Train Loop
def train():
    if optimizer_type == "Adam":
        optimizer = torch.optim.Adam(model.get_parameters(weight_decay), lr=learning_rate, weight_decay=weight_decay, betas=(b1, b2))
    elif optimizer_type == "AdamW":
        optimizer = torch.optim.AdamW(model.get_parameters(weight_decay), lr=learning_rate, weight_decay=weight_decay, betas=(b1, b2))
    elif optimizer_type == "SGD":
        optimizer = torch.optim.SGD(model.get_parameters(weight_decay), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)

    for epoch in range(epochs):
        loader = train_loader if not just_one else [next(iter(train_loader))]
        train_loss = train_one_epoch(model=model, loader=loader, optimizer=optimizer)
        run.log({"train/loss": train_loss})
        
        if epoch % valid_term == 0 and not just_one:
            eval_loss = evaluation(epoch)
            run.log({"val/loss": eval_loss})
        
        # SAVE the checkpoint
        if (epoch+1)%save_term == 0:
            torch.save(model.state_dict(), f"{checkpoint_dir}/{exp_name}/epoch-{epoch}.pt")


train()