import math
import torch
from torch import nn
import torch.nn.functional as F
from models.odconv import ODConv2d


class EnhancedDecoder(nn.Module):
    """Cross-layer dynamic convolution fusion decoder for masked query reconstruction."""

    def __init__(self, input_sizes=[256, 512, 1024, 2048], output_channels=1):
        super().__init__()
        self.feature_adapters = nn.ModuleList([
            self._build_adapter(in_c, 256) for in_c in input_sizes
        ])
        self.fusion_logits = nn.Parameter(torch.zeros(len(input_sizes)))
        self.fusion = nn.Conv2d(256, 256, kernel_size=1, stride=1, bias=False)
        self.reconstruct = nn.Sequential(
            ODResBlock(256, 256, kernel_num=4),
            UpsampleBlock(256, 128),
            ODResBlock(128, 64, kernel_num=2),
            UpsampleBlock(64, 32),
            nn.Conv2d(32, output_channels, 3, padding=1),
            nn.Sigmoid()
        )

    def _build_adapter(self, in_c, out_c):
        """Build the ODConv-based adapter for one encoder level."""
        layers = [
            ODConv2d(in_c, out_c, 3, padding=1, kernel_num=2),
            nn.BatchNorm2d(out_c),
            nn.GELU()
        ]
        if in_c == 2048:
            layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
            layers.append(ODConv2d(out_c, out_c, 3, padding=1, kernel_num=2))
            layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.GELU())
        return nn.Sequential(*layers)

    def forward(self, features):
        """Fuse multi-level reconstruction features and decode the masked query image."""
        adapted = [adpt(feat) for adpt, feat in zip(self.feature_adapters, features)]
        target_size = adapted[0].shape[-2:]
        adapted = [
            feat if feat.shape[-2:] == target_size else F.interpolate(feat, size=target_size, mode='bilinear', align_corners=True)
            for feat in adapted
        ]
        weights = torch.softmax(self.fusion_logits, dim=0)
        fused = sum(w * feat for w, feat in zip(weights, adapted))
        fused = self.fusion(fused)
        return self.reconstruct(fused)


class CrossAttention2D(nn.Module):
    """Cross-attention from segmentation features to reconstruction features."""

    def __init__(self, seg_channels, rec_channels, num_heads=8):
        super().__init__()
        assert seg_channels % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = seg_channels // num_heads
        self.q_proj = nn.Conv2d(seg_channels, seg_channels, kernel_size=1, bias=False)
        self.k_proj = nn.Conv2d(rec_channels, seg_channels, kernel_size=1, bias=False)
        self.v_proj = nn.Conv2d(rec_channels, seg_channels, kernel_size=1, bias=False)
        self.out_proj = nn.Conv2d(seg_channels, seg_channels, kernel_size=1, bias=False)

    def forward(self, seg_feat, rec_feat):
        """Project reconstruction priors to the segmentation feature space."""
        b, c, h, w = seg_feat.shape
        if rec_feat.shape[-2:] != (h, w):
            rec_feat = F.interpolate(rec_feat, size=(h, w), mode='bilinear', align_corners=True)

        q = self.q_proj(seg_feat).flatten(2).transpose(1, 2)
        k = self.k_proj(rec_feat).flatten(2).transpose(1, 2)
        v = self.v_proj(rec_feat).flatten(2).transpose(1, 2)

        q = q.view(b, h * w, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, h * w, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, h * w, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = torch.softmax(attn, dim=-1)
        z = torch.matmul(attn, v)
        z = z.transpose(1, 2).contiguous().view(b, h * w, c)
        z = z.transpose(1, 2).contiguous().view(b, c, h, w)
        return self.out_proj(z)


class UCI(nn.Module):
    """Uncertainty-gated cross-attention injection module."""

    def __init__(self, seg_channels, rec_channels, num_heads=8, kernel_size=5):
        super().__init__()
        self.cross_attn = CrossAttention2D(seg_channels, rec_channels, num_heads=num_heads)
        self.gate_conv = nn.Conv2d(3, 1, kernel_size=1, bias=True)
        self.align = nn.Conv2d(seg_channels, seg_channels, kernel_size=1, bias=False)
        self.kernel_size = kernel_size

    def _boundary_band(self, prob):
        """Construct the boundary band from the probability map."""
        k = self.kernel_size
        pad = k // 2
        dil = F.max_pool2d(prob, kernel_size=k, stride=1, padding=pad)
        ero = 1.0 - F.max_pool2d(1.0 - prob, kernel_size=k, stride=1, padding=pad)
        return (dil - ero).clamp(0.0, 1.0)

    def _edge_strength(self, feat):
        """Estimate edge strength from the segmentation feature map."""
        grad_x = feat[:, :, :, 1:] - feat[:, :, :, :-1]
        grad_y = feat[:, :, 1:, :] - feat[:, :, :-1, :]
        grad_x = F.pad(grad_x, (0, 1, 0, 0))
        grad_y = F.pad(grad_y, (0, 0, 0, 1))
        mag = torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-6)
        return mag.mean(dim=1, keepdim=True)

    def forward(self, seg_feat, rec_feat, prob):
        """Inject reconstruction priors into uncertain or boundary-sensitive regions."""
        if prob.dim() == 3:
            prob = prob.unsqueeze(1)
        if prob.shape[-2:] != seg_feat.shape[-2:]:
            prob = F.interpolate(prob, size=seg_feat.shape[-2:], mode='bilinear', align_corners=True)
        prob = prob.clamp(1e-6, 1 - 1e-6)

        uncertainty = (-prob * torch.log(prob) - (1 - prob) * torch.log(1 - prob)) / math.log(2.0)
        boundary = self._boundary_band(prob)
        edge = self._edge_strength(seg_feat)
        gate = torch.sigmoid(self.gate_conv(torch.cat([uncertainty, boundary, edge], dim=1)))

        z = self.cross_attn(seg_feat, rec_feat)
        fused = seg_feat + gate * z
        return fused + self.align(fused)


class ODResBlock(nn.Module):
    """ODConv residual block used in the reconstruction decoder."""

    def __init__(self, in_c, out_c, expansion=2, kernel_num=2):
        super().__init__()
        hidden_c = in_c * expansion
        self.conv = nn.Sequential(
            ODConv2d(in_c, hidden_c, 3, padding=1, kernel_num=kernel_num),
            nn.BatchNorm2d(hidden_c),
            nn.GELU(),
            ODConv2d(hidden_c, out_c, 3, padding=1, kernel_num=kernel_num),
            nn.BatchNorm2d(out_c)
        )
        self.shortcut = ODConv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()
        self.channel_attn = ChannelAttention(out_c)

    def forward(self, x):
        """Apply residual reconstruction with channel attention."""
        return self.channel_attn(self.conv(x) + self.shortcut(x))


class ChannelAttention(nn.Module):
    """Efficient channel attention used inside reconstruction blocks."""

    def __init__(self, channels, reduction=8):
        super().__init__()
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.GELU(),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """Reweight channels using global context."""
        return x * self.gate(x)


class UpsampleBlock(nn.Module):
    """Learnable upsampling block for progressive reconstruction."""

    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ODConv2d(in_c, out_c, 3, padding=1, kernel_num=2),
            nn.BatchNorm2d(out_c),
            nn.GELU()
        )

    def forward(self, x):
        """Upsample and refine the reconstruction feature map."""
        return self.conv(x)
