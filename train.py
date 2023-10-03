import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchmetrics impt functional as FM
import pytorch_lightning as pl
from lightning.pytorch.loggers import WandbLogger

from argparse import ArgumentParser

from braincoder import linear
from dataloader import create_dataloader
from braincoder import generate, prepare_diffuser, prepare_text_embedding

# Get SYS arguments of model type selection.
parser = ArgumentParser(description="braincoder training pipeline")

deafult_model_name = "linear"
parser.add_argument("--model_name", default=deafult_model_name, type=str)

args = parser.parse_args()
model_name = args.model_name

# ================= HYPER PARAMETERS ================
models = {
    "linear": linear,
    "coatnet": None
}

learning_rate = 1e-3
batch_size = 64
epochs = 20

cache_dir = "./cache.hdf5"

# =================== DATASET LOADING =====================
train_loader, eval_loader = create_dataloader(cache_dir=cache_dir)

# ================== LOGGER =================
wandb_logger = WandbLogger(project="braincoder")


# ================ Pytorch Ligthning Training Module ===========
class LigthningPipeline(pl.LightningModule):
    def __init__(self, model_name, lr, device, **kwargs):
        super().__init__()
        self.learning_rate = lr
        self.device = device

        self.model_name = model_name
        self.model = models[model_name](**kwargs)

        self.criterion = nn.KLDivLoss(reduction="batchmean")

        self.vae, self.unet, self.scheduler = prepare_diffuser(device=device)
        self.tokenizer, self.text_encoder = prepare_text_embedding(device=device)
        
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

    # def validation_step(self, batch, batch_idx):
    #     x, y = batch
    #     logits = self(x)
    #     loss = F.l1_loss(logits, y)
    #     metrics = {'val_loss': loss}
    #     self.log_dict(metrics)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.l1_loss(logits, y)
        metrics = {'test_loss': loss}
        self.log_dict(metrics)    

    def diffuse_with_braincoder_emb(self, emb):
        pil_images = generate(embedding=emb, vae=self.vae, unet=self.unet, scheduler=self.scheduler, device=self.device)

        return pil_images

    def configure_optimizers(self):
        try:
            param = self.model.get_parameters()
        except: 
            param = self.model.parameters()

        return torch.optim.Adam(param, lr=self.learning_rate)


model = LigthningPipeline(model_name, learning_rate)

# Logging Gradient
wandb_logger.watch(model)

# Lightning Trainer for Easy Pipeline Constructing
trainer = pl.Trainer(max_epochs=epochs, gpus=1)
trainer.fit(model, train_loader)

trainer.test(eval_loader)

model.eval()
### Evaluation step