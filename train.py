import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchmetrics impt functional as FM
import pytorch_lightning as pl
from lightning.pytorch.loggers import WandbLogger

from argparse import ArgumentParser

from model import LinearModel
from dataloader import create_dataloader
from diffusion_helper import generate, prepare_diffuser, prepare_text_embedding

# Get SYS arguments of model type selection.
parser = ArgumentParser(description="braincoder training pipeline")

deafult_model_name = "linear"
parser.add_argument("--model_name", default=deafult_model_name, type=str)

args = parser.parse_args()
model_name = args.model_name

# ================= HYPER PARAMETERS ================
models = {
    "linear": LinearModel,
    "coatnet": None
}

device = "cuda" if torch.cuda.is_available() else "cpu"

learning_rate = 1e-3
batch_size = 64
epochs = 20

b1, b2 = 0.95, 0.99

cache_dir = "./cache.hdf5"

checkpoint_dir = f"./tmp/{model_name}"

# =================== DATASET LOADING =====================
train_loader, eval_loader = create_dataloader(batch_size=batch_size, cache_dir=cache_dir)

num_to_samples = 5
to_samples = []
to_samples_keys = []
for i in range(num_to_samples):
    temp, key = next(iter(eval_loader))
    to_samples.append(temp)
    to_samples_keys.append(key)
to_samples = torch.stack(to_samples, dim=0).to(device)
#to_sample = next(iter(eval_loader))

# ================== LOGGER =================
wandb_logger = WandbLogger(project="braincoder")


# ------------------ Prepare DIFFUSION GUYS ---------------------
vae, unet, scheduler = prepare_diffuser(device=device)
tokenizer, text_encoder = prepare_text_embedding(device=device)


# ================ Pytorch Ligthning Training Module ===========
class LigthningPipeline(pl.LightningModule):
    def __init__(self, model_name, to_samples, to_sample_keys, lr, device, batch_size, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = lr
        self.device = device
        self.batch_size = batch_size

        self.model_name = model_name
        self.model = models[model_name](**kwargs)

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


model = LigthningPipeline(model_name=model_name, learning_rate=learning_rate, batch_size=batch_size, to_samples=to_samples, to_sample_keys=to_samples_keys, device=device)

# Logging Gradient
wandb_logger.watch(model)

# Lightning Trainer for Easy Pipeline Constructing
trainer = pl.Trainer(max_epochs=epochs, gpus=1, default_root_dir=checkpoint_dir)
trainer.fit(model, train_loader, eval_loader)
