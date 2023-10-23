import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchmetrics impt functional as FM
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb
import tqdm

from argparse import ArgumentParser
import os
import time
from pathlib import Path
# from dataclasses import Union

from model import LinearModel, CoAtNet
from dataloader import create_dataloader
from diffusion_helper import generate, prepare_diffuser, prepare_text_embedding
from utils import read_config

# Get SYS arguments of model type selection.
parser = ArgumentParser(description="braincoder training pipeline")

deafult_model_name = "linear"
parser.add_argument("--model_name", default=deafult_model_name, type=str)
parser.add_argument("--cfg", default="./config/exp_config_yaml", type=str)

args = parser.parse_args()
model_name = args.model_name
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
checkpoint_dir = exp_cfg["checkpoint_dir"]
# os.mkdir(checkpoint_dir)
# os.makedirs(checkpoint_dir, exist_ok=True)
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

valid_term = 1

print(cfg)

# =================== DATASET LOADING =====================
train_loader, eval_loader = create_dataloader(batch_size=batch_size, cache_dir=cache_dir, image_dir=image_dir, device=device)

to_samples = []
to_samples_keys = []
# for i in range(num_to_samples):
temp, _, key = next(iter(eval_loader))
to_samples = temp[:num_to_samples].to("cpu")
to_samples_keys = key[:num_to_samples]
print(to_samples_keys)
# to_samples = torch.stack(to_samples, dim=0).to(device)
#to_sample = next(iter(eval_loader))

# ================== LOGGER =================
# wandb_logger = WandbLogger(project="braincoder")
model = models[model_name].from_cfg(model_cfg).to(device)

total_parameters = model.num_parameters()

run = wandb.init(
  project="braincoder",
  config={**model_cfg,**exp_cfg, "param": total_parameters}
)

# ------------------ Prepare DIFFUSION GUYS ---------------------
# vae, unet, scheduler = prepare_diffuser(device=device)
# tokenizer, text_encoder = prepare_text_embedding(device=device)


# ---------------- VANILLA TRaining LOOP --------------------
# compiled_model = torch.compile(model)
# are you criminal?
# wandb.watch(model, log="gradients")

def loss_term(y, y_hat):
    # Base LOSS will be L2
    loss = alpha*F.mse_loss(y_hat, y)

    if "kl" in metrics:
        loss = loss + (1-alpha)*F.kl_div(F.softmax(y_hat), F.softmax(y))
    elif "contrastive" in metrics:
        # loss = loss + (1-alpha)*
        pass
    elif "cross_en":
        loss = loss + (1-alpha)*F.cross_entropy(y_hat, y)

    return loss

def train():
    if optimizer_type == "Adam":
        optimizer = torch.optim.Adam(model.get_parameters(weight_decay), lr=learning_rate, weight_decay=weight_decay, betas=(b1, b2))
    elif optimizer_type == "AdamW":
        optimizer = torch.optim.AdamW(model.get_parameters(weight_decay), lr=learning_rate, weight_decay=weight_decay, betas=(b1, b2))
    elif optimizer_type == "SGD":
        optimizer = torch.optim.SGD(model.get_parameters(weight_decay), lr=learning_rate, weight_decay=weight_decay, momentume=0.9)

    for epoch in range(epochs):
        # Main Training
        _loss = []
        for step, batch in enumerate(tqdm.tqdm(train_loader)):
            x, y, _ = batch
            yhat = model(x)

            loss = loss_term(y, yhat)
            loss /= grad_accum

            loss.backward()
            _loss.append(loss.item())

            # Gradient Accumulation hahahahahahahahahahhaha I need just A100 
            if (step+1)%grad_accum==0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip, norm_type=2)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                
        # _loss.append(loss.cpu().detach().item())
        # run.log({"train/loss": _loss.sum() / len(_loss)})
        run.log({"train/loss": sum(loss) / len(loss), "epoch": epoch})
        
        if epoch % valid_term == 0:
            with torch.no_grad():
                _loss = []
                for batch in eval_loader:
                    x, y, im_keys = batch
                    yhat = model(x)
                    loss = loss_term(y, yhat)
                    
                    # Saving sample for visualizing the result
                    for i in range(how_many_to_save):
                        pred = yhat[i].cpu().detach()
                        im_key = im_keys[i].item()
                        filename = f"{checkpoint_dir}/{exp_name}/sample-{epoch}-{im_key}.pt"
                        torch.save(pred, filename)

                    _loss.append(loss.cpu().detach().item())

                run.log({"val/loss": sum(_loss) / len(_loss), "epoch": epoch})

        torch.save(model.state_dict(), f"{checkpoint_dir}/{exp_name}/epoch-{epoch}.pt")
    


train()

# ================ Pytorch Ligthning Training Module ===========
class LigthningPipeline(pl.LightningModule):
    def __init__(self, model_name, to_samples, to_sample_keys, learning_rate, device, batch_size, alpha, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        # self.device = device
        self.batch_size = batch_size
        self.alpha = alpha

        self.model_name = model_name
        self.model = models[model_name].from_cfg(**kwargs)

        self.criterion = nn.KLDivLoss(reduction="batchmean")

        self.to_samples = to_samples
        self.to_sample_keys = to_sample_keys

    def forward(self, x):
        # if self.model_name in ["cnn", "attention"]:
        #     # for cnn input process is required
        #     # (B, 14, 640, 480, 3)
        #     # (B, 14*3, 640, 480)
        #     # consider eeg channel with image channel
        #     # experiment the effect of these concat(?)
        #     pass
        return self.model(x)

    def loss_term(self, y_hat, y):
        l2_loss = F.mse_loss(y_hat, y)
        kl_loss = self.criterion(F.softmax(y_hat), F.softmax(y))
        loss = self.alpha*l2_loss + (1-self.alpha)*kl_loss

        return loss.item()

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self(x)
        loss = self.loss_term(y_hat, y)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self(x)
        loss = F.mse_loss(logits, y).cpu().detach().item()
        #metrics = {'test_loss': loss}
        #self.log_dict(metrics)    
        self.log("test/loss", loss)

    def diffuse_with_braincoder_emb(self, emb):
        pil_images = generate(embedding=emb, vae=self.vae, unet=self.unet, scheduler=self.scheduler, device=self.device)

        return pil_images

    def configure_optimizers(self):
        try:
            param = self.model.get_parameters()
        except: 
            param = self.model.parameters()

        return torch.optim.Adam(param, lr=self.learning_rate, betas=(b1, b2))

    # def on_epoch_end(self):
    #     z = self.model(self.to_samples)

    #     images = self.diffuse_with_braincoder_emb(z)

    #     wandb_logger.log_image(key="samples", images=images, caption=self.to_sample_keys)


# model = LigthningPipeline(model_name=model_name, learning_rate=learning_rate, batch_size=batch_size, to_samples=to_samples, to_sample_keys=to_samples_keys, device=device, alpha=alpha, cfg=model_cfg)

# Logging Gradient
# wandb_logger.watch(model)

# Lightning Trainer for Easy Pipeline Constructing
# trainer = pl.Trainer(max_epochs=epochs, accelerator="gpu", devices=1, default_root_dir=checkpoint_dir, num_sanity_val_steps=0, enable_progress_bar=True)
# trainer.fit(model, train_loader, eval_loader)
