import torch
from torch import nn


class FreqBranchCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(12, 24, kernel_size=3, padding=1, groups=12)
        self.conv2 = nn.Conv1d(24, 24, kernel_size=3, padding=1, groups=12)
        self.proj = nn.Conv1d(24, 12, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(24)
        self.bn2 = nn.BatchNorm1d(24)
        self.bn3 = nn.BatchNorm1d(12)
        self.act = nn.GELU()
        self.drop = nn.Dropout(0.15)
        self.channel_proj = nn.Linear(11, 10)

    def forward(self, values):
        hidden = self.act(self.bn1(self.conv1(values)))
        hidden = self.act(self.bn2(self.conv2(hidden)))
        hidden = self.act(self.bn3(self.proj(hidden)))
        return self.channel_proj(self.drop(hidden))


class StochasticDepth(nn.Module):
    def __init__(self, drop_prob=0.1):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, values):
        if not self.training or self.drop_prob == 0:
            return values
        keep_probability = 1 - self.drop_prob
        shape = (values.shape[0],) + (1,) * (values.ndim - 1)
        mask = torch.empty(shape, dtype=values.dtype, device=values.device)
        mask.bernoulli_(keep_probability).div_(keep_probability)
        return values * mask


class DualAxisAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.feat_attn = nn.TransformerEncoderLayer(
            d_model=12, nhead=4, dim_feedforward=48, dropout=0.15,
            activation='gelu', batch_first=True, norm_first=True)
        self.stoch_depth = StochasticDepth(0.1)
        self.chan_mix = nn.Linear(12, 12, bias=False)
        nn.init.eye_(self.chan_mix.weight)
        self.chan_norm = nn.LayerNorm(128)
        self.chan_pos = nn.Parameter(torch.randn(1, 12, 128) * 0.02)

    def forward(self, values):
        transposed = values.transpose(1, 2)
        attended = self.feat_attn(transposed)
        transposed = transposed + self.stoch_depth(attended - transposed)
        positioned = transposed.transpose(1, 2) + self.chan_pos
        mixed = self.chan_mix(positioned.transpose(1, 2)).transpose(1, 2)
        return self.chan_norm(positioned + mixed)


class DSFF(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()
        self.low_branch = FreqBranchCNN()
        self.high_branch = FreqBranchCNN()
        self.freq_drop = nn.Dropout2d(0.2)
        self.channel_proj = nn.Sequential(nn.Linear(20, 128), nn.LayerNorm(128), nn.GELU())
        self.dual_axis_attn = DualAxisAttention()
        self.head = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, 64), nn.GELU(),
                                  nn.Dropout(0.15), nn.Linear(64, num_classes))

    def forward(self, values):
        channels = values.reshape(-1, 12, 22)
        low = self.low_branch(channels[:, :, :11])
        high = self.freq_drop(channels[:, :, 11:].unsqueeze(-1)).squeeze(-1)
        high = self.high_branch(high)
        fused = self.channel_proj(torch.cat([low, high], dim=-1))
        attended = self.dual_axis_attn(fused)
        return self.head(attended.mean(dim=1))
