import torch
import torch.nn as nn
import torch.nn.functional as F

from argparse import ArgumentParser

import optuna
from optuna.trial import TrialState

from model import LinearModel, CoAtNet
from dataloader import create_dataloader



models = {
    "linear": LinearModel,
    "coatnet": CoAtNet
}
device = "cuda" if torch.cuda.is_available() else "cpu"


train_loader, eval_loader = create_dataloader(batch_size=10, cache_dir= "/content/drive/MyDrive/brainstormers/cache.hdf5", scale=0.01, image_dir="/content/dataset", device=device)


study = optuna.create_study(direction="minimize")

# Define Model with Optuna Trial
def define_model(trial):
    dropout = trial.suggest_float("dropout",1e-2, 1e-1, log=True)
    third_layer = trial.suggest_int("third_layer", 1, 3)
    forth_layer = trial.suggest_int("forth_layer", 1, 4)
    model_cfg = {"image_shape": [320, 240], "initial_channel": 42, "num_blocks": [2, 2, third_layer, forth_layer], "channels": [42, 64, 196, 384, 768], "block_type": ["C", "C", "T", "T"], "dropout": dropout}
    model = models["coatnet"].from_cfg(model_cfg).to(device)

    return model


def loss_term(y, y_hat, alpha):
    mse_loss = ((y_hat-y)**2).mean()
    cos_loss = 1 - torch.cosine_similarity(y_hat, y, dim=-1).mean()
    loss = alpha*mse_loss + (1-alpha)*cos_loss

    return loss

def objective(trial):
    model = define_model(trial)

    optimizer_type = trial.suggest_categorical("optimizer", ["Adam", "AdamW", "SGD"])
    learning_rate = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    weight_decay = trial.suggest_float("weight_decay", 0.001, 0.1)
    alpha = 0.9
    grad_clip = trial.suggest_int("grad_clip", 0, 5)

    if optimizer_type == "Adam":
        optimizer = torch.optim.Adam(model.get_parameters(weight_decay), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type == "AdamW":
        optimizer = torch.optim.AdamW(model.get_parameters(weight_decay), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type == "SGD":
        optimizer = torch.optim.SGD(model.get_parameters(weight_decay), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)

    batch = next(iter(train_loader))
    for _ in range(100):
        x, y, im_key = batch
        yhat = model(x)

        loss = loss_term(y, yhat, alpha)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip, norm_type=2)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    return loss.item()

study.optimize(objective, n_trials=200, timeout=60000000)

pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

print("Study statistics: ")
print("  Number of finished trials: ", len(study.trials))
print("  Number of pruned trials: ", len(pruned_trials))
print("  Number of complete trials: ", len(complete_trials))

print("Best trial:")
trial = study.best_trial

print("  Value: ", trial.value)

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))