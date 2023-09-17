import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import functional as FM
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core.lightning import LightningModule

from LinearModel import LinearModel

models = {
    "linear": LinearModel,
    "cnn": None,
    "attention": None
}

class LinearModel(LightningModule):
    def __init__(self, model_name, lr,**kwargs):
        super().__init__()

        self.learning_rate = lr
        self.model = models[model_name](**kwargs)
        self.criterion = nn.KLDivLoss(reduction="batchmean")
        
    def forward(self, x):
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
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.l1_loss(logits, y)
        metrics = {'val_loss': loss}
        self.log_dict(metrics)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.l1_loss(logits, y)
        metrics = {'test_loss': loss}
        self.log_dict(metrics)    

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
