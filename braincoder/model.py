import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
import math

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


def conv_3x3_bn(inp, oup, image_shape, downsample=False):
    stride = 1 if downsample == False else 2
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup), 
        nn.GELU()
    )

class PreNorm(nn.Module):
    def __init__(self, dim, fn, norm):
        super().__init__()
        self.norm = norm(dim)
        self.fn = fn
    
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class SE(nn.Module):
    def __init__(self, inp, oup, expansion=0.25):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # rank reduction for feature extraction(?)
        self.fc = nn.Sequential(
            nn.Linear(oup, int(inp * expansion), bias=False),
            nn.GELU(),
            nn.Linear(int(inp * expansion), oup, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c) # [b, c]
        y = self.fc(y).view(b, c, 1, 1) # [b, c, 1, 1]
        return x * y

class MBConv(nn.Module):
    def __init__(self, inp, oup, image_shape, downsample=False, **kwargs):
        super().__init__()
        self.downsample = downsample
        stride = 1 if self.downsample == False else 2
        hidden_dim = int(inp * 4)

        if self.downsample:
            # pooling for rescon
            self.pool = nn.MaxPool2d(3, 2, 1)
            self.proj = nn.Conv2d(inp, oup, 1, 1, 0, bias=False)

        self.conv = nn.Sequential(
            # down-sample in the first conv
            nn.Conv2d(inp, hidden_dim, 1, stride, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            SE(inp, hidden_dim),
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup)
        )

        self.conv = PreNorm(inp, self.conv, nn.BatchNorm2d)

    def forward(self, x):
        # downsamplling
        if self.downsample:
            return self.proj(self.pool(x)) + self.conv(x)
        
        # residual connection
        return x + self.conv(x)

class Attention(nn.Module):
    def __init__(self, inp, oup, image_size, heads=8, dim_head=32, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        # projection disablinng when casual attention
        projection = not (heads == 1 and dim_head == inp)

        self.ih, self.iw = image_size

        self.heads = heads
        self.scale = dim_head ** -0.5

        # paramter table of relative position bias
        self.relative_bias_table = nn.Parameter(
            torch.zeros((2*self.ih-1)*(2*self.iw-1), heads),
        )

        coords = torch.meshgrid((torch.arange(self.ih), torch.arange(self.iw)))
        coords = torch.flatten(torch.stack(coords), 1)
        relative_coords = coords[:, :, None] - coords[:, None, :]

        relative_coords[0] += self.ih - 1
        relative_coords[1] += self.iw - 1
        relative_coords[0] *= 2 * self.iw - 1
        relative_coords = rearrange(relative_coords, "c h w -> h w c")
        relaitve_index = relative_coords.sum(-1).flatten().unsqueeze(1)
        self.register_buffer("relative_index", relaitve_index)

        # b c ih iw
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(inp, inner_dim * 3, bias=False)
        
        self.out_proj = nn.Sequential(
            nn.Linear(inner_dim, oup),
            nn.Dropout(dropout),
        ) if projection else nn.Identity()

    def forward(self, x):
        # multihead attention implementation
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(
            t, "b n (h d) -> b h n d", h=self.heads
        ), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # User "gather" for more efficiency on GPUs
        relative_bias = self.relative_bias_table.gather(
            0, self.relative_index.repeat(1, self.heads)
        )
        relative_bias = rearrange(
            relative_bias, "(h w) c -> 1 c h w", h=self.ih*self.iw, w=self.ih*self.iw
        )
        dots = dots + relative_bias

        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.out_proj(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)

class Transformer(nn.Module):
    def __init__(self, inp, oup, image_size, heads=8, dim_head=32, downsample=False, dropout=0.):
        super().__init__()
        hidden_dim = int(inp * 4)
        self.ih, self.iw = image_size
        self.downsample = downsample

        if downsample:
            self.pool1 = nn.MaxPool2d(3, 2, 1)
            self.pool2 = nn.MaxPool2d(3, 2, 1)
            self.proj = nn.Conv2d(inp, oup, 1, 1, 0, bias=False)

        self.attn = Attention(inp, oup, image_size, heads, dim_head, dropout)
        self.ff = FeedForward(oup, hidden_dim, dropout)

        self.attn = nn.Sequential(
            Rearrange("b c ih iw -> b (ih iw) c"),
            PreNorm(inp, self.attn, nn.LayerNorm),
            Rearrange("b (ih iw) c -> b c ih iw", ih=self.ih, iw=self.iw)
        )

        self.ff = nn.Sequential(
            Rearrange("b c ih iw -> b (ih iw) c"),
            PreNorm(oup, self.ff, nn.LayerNorm),
            Rearrange("b (ih iw) c -> b c ih iw", ih=self.ih, iw=self.iw)
        )

    def forward(self, x):
        if self.downsample:
            x = self.proj(self.pool1(x)) + self.attn(self.pool2(x))
        else:
            x = x + self.attn(x)
        x = x + self.ff(x)
        return x


class CoAtNet(nn.Module):
    CLIP_SHAPE = (77, 768)
    def __init__(self, image_shape, num_blocks, channels, block_type=["C", "C", "T", "T"], dropout=0., training=True):
        super().__init__()
        ih, iw = image_shape

        # MBConv for utilizing convolution's inductive bias
        # Transformer ViT for large model ability
        block = {"C": MBConv, "T": Transformer}

        self.layers = nn.ModuleList([])
        for i in range(len(block_type)):
            _image_shape = (ih // (2**(i+1)), iw // (2**(i+1)))
            _s = self._make_layer(block[block_type[i]], channels[i], channels[i+1], num_blocks[i], _image_shape, dropout=dropout)
            self.layers.append(_s)
    
        factor = 2**len(block_type)
        out = (ih // factor) * (iw // factor)
        self.proj = nn.Sequential(
            Rearrange("b c ih iw -> b c (ih iw)"),
            nn.Linear(out, CoAtNet.CLIP_SHAPE[0]), 
            Rearrange("b c i -> b i c")
        )

        if training:
            self.apply(self._init_weights)
            
            for pn, p in self.named_parameters():
                if pn.endswith('proj.1.weight'):
                    torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(sum(num_blocks)))

    def forward(self, x):
        for s in self.layers:
            x = s(x)
        x = self.proj(x)
        return x

    def _make_layer(self, block, inp, oup, depth, image_size, **kwargs):
        layers = nn.ModuleList([])
        for d in range(depth):
            if d == 0:
                layers.append(block(inp, oup, image_size, downsample=True, **kwargs))
            else:
                layers.append(block(oup, oup, image_size, **kwargs))
        return nn.Sequential(*layers)
    
    def get_parameters(self, weight_decay):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (nn.BatchNorm2d)
        for mn, m in self.named_parameters():
            # print(mn)
            if mn.endswith("bias"):
                no_decay.add(mn)
            elif mn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                decay.add(mn)
            elif mn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                no_decay.add(mn)
            else:
                decay.add(mn)

        param_dict = {pn: p for pn, p in self.named_parameters()}

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0}
        ]

        return optim_groups

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def num_parameters(self):
        return sum([p.nelement() for p in self.parameters()])

    @classmethod
    def from_cfg(cls, cfg):
        return cls(cfg["image_shape"], cfg["num_blocks"], cfg["channels"], cfg["block_type"], dropout=cfg["dropout"])

    @classmethod
    def from_pretrained(cls, cfg, path, finetune=False):
        model = cls.from_cfg(cfg)
        param = torch.load(path)
        model.load_state_dict(param)

        # Train only last projection layer on fine-tuning process.
        if finetune:
            for p, pn in model.named_parameters():
                if not pn.endswith('proj.1.weight'):
                    p.requires_grad = False

        return model


if __name__ == "__main__":
    from utils import read_config
    cfg = read_config("./config/exp_config.yaml")
    network = CoAtNet.from_cfg(cfg["model"])
    network.get_parameter(0.1)
    print(network.num_paramteres())

    a = torch.zeros((1, 42, 320, 240))
    network(a)
