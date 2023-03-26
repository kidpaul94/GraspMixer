import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class MixerBlock(nn.Module):
    def __init__(self, dim: int, num_patch: int, token_dim: int, channel_dim: int, dropout: float = 0.):
        super(MixerBlock, self).__init__()
        self.token_mix = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n d -> b d n'),
            FeedForward(num_patch, token_dim, dropout),
            Rearrange('b d n -> b n d')
        )
        self.channel_mix = nn.Sequential(
            nn.LayerNorm(dim),
            FeedForward(dim, channel_dim, dropout),
        )

    def forward(self, x):
        x = x + self.token_mix(x)
        x = x + self.channel_mix(x)

        return x

class SimpleMixer(nn.Module):
    def __init__(self, dim: int, num_patch: int, embed_dim: int, depth: int, token_dim: int, 
                 channel_dim: int, num_classes: int):
        super(SimpleMixer, self).__init__()
        self.embedding = nn.Sequential(nn.Linear(dim, embed_dim))
        mixers = [MixerBlock(embed_dim, num_patch, token_dim, channel_dim) for _ in range(depth)]
        self.backbone = nn.Sequential(*mixers)
        self.mlp_head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, num_classes))

    def forward(self, x):
        x = self.embedding(x)
        x = self.backbone(x)
        x = self.mlp_head(x)

        return x
    
class LLGBlock(nn.Module):
    def __init__(self, dim: int, embed_dim: int):
        super(LLGBlock, self).__init__()
        self.embedding = nn.Sequential(
            nn.Linear(dim, embed_dim), 
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )

    def forward(self, x):
        x = self.embedding(x)

        return x
    
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.llg = LLGBlock(dim=3, embed_dim=36)
        self.mixer = SimpleMixer(dim=36, num_patch=207, embed_dim=128, depth=4, token_dim=128, channel_dim=1024,
                                 num_classes=10)

    def forward(self, x):
        proj = self.llg(x[1])
        x = torch.cat((x[0], proj), 1)
        x = self.mixer(x)

        return x
