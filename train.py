import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchmetrics impt functional as FM
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from argparse import ArgumentParser
import os

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

b1, b2 = exp_cfg["betas"]

cache_dir = exp_cfg["cache_dir"]
checkpoint_dir = exp_cfg["checkpoint_dir"]
# os.mkdir(checkpoint_dir)
os.makedirs(checkpoint_dir, exist_ok=True)
image_dir = exp_cfg["image_dir"]

num_to_samples = exp_cfg["num_to_samples"]

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
wandb_logger = WandbLogger(project="braincoder")


# ------------------ Prepare DIFFUSION GUYS ---------------------
vae, unet, scheduler = prepare_diffuser(device=device)
tokenizer, text_encoder = prepare_text_embedding(device=device)


# ================ Pytorch Ligthning Training Module ===========
class LigthningPipeline(pl.LightningModule):
    def __init__(self, model_name, to_samples, to_sample_keys, learning_rate, device, batch_size, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        # self.device = device
        self.batch_size = batch_size

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
        l1_loss = F.l1_loss(y_hat, y)
        kl_loss = self.criterion(y_hat, y)
        loss = l1_loss + kl_loss

        return loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_term(y_hat, y)
        self.log({"train/loss", loss})
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.l1_loss(logits, y)
        #metrics = {'test_loss': loss}
        #self.log_dict(metrics)    
        self.log({"test/loss", loss})

    def diffuse_with_braincoder_emb(self, emb):
        pil_images = generate(embedding=emb, vae=self.vae, unet=self.unet, scheduler=self.scheduler, device=self.device)

        return pil_images

    def configure_optimizers(self):
        try:
            param = self.model.get_parameters()
        except: 
            param = self.model.parameters()

        return torch.optim.Adam(param, lr=self.learning_rate, betas=(b1, b2))

    def on_epoch_end(self):
        z = self.model(self.to_samples)

        images = self.diffuse_with_braincoder_emb(z)

        wandb_logger.log_image(key="samples", images=images, caption=self.to_sample_keys)


model = LigthningPipeline(model_name=model_name, learning_rate=learning_rate, batch_size=batch_size, to_samples=to_samples, to_sample_keys=to_samples_keys, device=device, cfg=model_cfg)

# Logging Gradient
wandb_logger.watch(model)

# Lightning Trainer for Easy Pipeline Constructing
trainer = pl.Trainer(max_epochs=epochs, accelerator="gpu", devices=1, default_root_dir=checkpoint_dir)
trainer.fit(model, train_loader, eval_loader)
