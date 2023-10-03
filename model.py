import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearModel(nn.Module):
    def __init__(self, eeg_channels, w=640, h=480, channels=3, hidden=[]):
        super().__init__()  
        assert hidden[-1] == 77*768, "Please check the hidden state size."

        self.eeg_c = eeg_channels
        self.w = w
        self.h = h
        self.c = channels

        self.initial_linear = nn.Linear(channels*w*h, hidden[0])
        layers = []
        for (fan_in, fan_out) in zip(hidden, hidden[1:]):
            layers.append(nn.Linear(fan_in, fan_out))

        self.linears = nn.ModuleList(**layers)        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # (Batch, 14, 640, 480, 3)
        B = x.size(0)
        x = x.view(B, self.eeg_c, -1)

        x = self.relu(self.initial_linear(x))
    
        for layer in self.layers:
            x = layer(x)
            x = self.relu(x)

        x = x.view(B, 77, 768)

        return x