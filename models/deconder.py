import torch
from torch import nn
from models.odconv import ODConv2d


class EnhancedDecoder(nn.Module):
    def __init__(self, input_sizes=[256, 512, 1024, 2048], output_channels=1):
        super().__init__()

        # 特征适配器：统一各层通道和分辨率
        self.feature_adapters = nn.ModuleList([
            self._build_adapter(in_c, 256) for in_c in input_sizes
        ])
        self.fusion = nn.Conv2d(256 * 4, 256, kernel_size=1, stride=1, bias=False)  # reduce feature map dimension
        # 渐进重建模块（修正通道维度）
        self.reconstruct = nn.Sequential(
            ODResBlock(256, 256, kernel_num=4),  # 输入通道对齐
            UpsampleBlock(256, 128),  # 64x64 → 128x128
            ODResBlock(128, 64, kernel_num=2),
            UpsampleBlock(64, 32),  # 128x128 → 256x256
            nn.Conv2d(32, output_channels, 3, padding=1),
            nn.Sigmoid()
        )

    def _build_adapter(self, in_c, out_c):
        layers = [
            ODConv2d(in_c, out_c, 3, padding=1, kernel_num=2),
            nn.BatchNorm2d(out_c),
            nn.GELU()
        ]
        # 仅对layer4特征进行上采样
        if in_c == 2048:
            layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
            layers.append(ODConv2d(out_c, out_c, 3, padding=1, kernel_num=2))
        return nn.Sequential(*layers)

    def forward(self, features):
        # 特征预处理
        adapted = [adpt(feat) for adpt, feat in zip(self.feature_adapters, features)]

        concat_features = torch.cat(adapted, dim=1)

        fused = self.fusion(concat_features)

        # 图像重建
        return self.reconstruct(fused)

#############20250704

class ChannelAttention_1(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        attn = self.avg_pool(x)
        attn = self.fc(attn)
        return x * attn


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # [B,1,H,W]
        attn = self.conv(avg_out)
        return x * self.sigmoid(attn)


class SemanticConsistencyFusion(nn.Module):
    def __init__(self, channels_dict):
        super().__init__()
        self.attn_blocks = nn.ModuleDict()

        for layer_name, c in channels_dict.items():
            self.attn_blocks[layer_name] = nn.Sequential(
                ChannelAttention_1(c),
                SpatialAttention()
            )
        # 可学习的融合系数 γ，每层一个
        self.gamma = nn.ParameterDict({
            layer: nn.Parameter(torch.zeros(1)) for layer in channels_dict
        })

    def forward(self, seg_fts, rec_fts):
        """
        seg_fts: dict of segmentation features from segmentation branch
        rec_fts: dict of reconstruction features from reconstruction branch
        Returns: updated segmentation features dict
        """
        out = {}
        for layer_name in seg_fts:
            seg_feat = seg_fts[layer_name]                 # [B, C, H, W]
            rec_feat = rec_fts[layer_name]                 # [B, C, H, W]
            attn_feat = self.attn_blocks[layer_name](rec_feat)  # apply attention
            fused = seg_feat + self.gamma[layer_name] * attn_feat
            out[layer_name] = fused
        return out

class ODResBlock(nn.Module):
    """动态卷积残差块（优化梯度流）"""

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

        self.shortcut = (
            ODConv2d(in_c, out_c, 1)
            if in_c != out_c
            else nn.Identity()
        )
        self.channel_attn = ChannelAttention(out_c)

    def forward(self, x):
        return self.channel_attn(self.conv(x) + self.shortcut(x))

class ChannelAttention(nn.Module):
    """高效通道注意力（优化计算效率）"""

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
        return x * self.gate(x)


class UpsampleBlock(nn.Module):
    """可学习上采样模块"""

    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ODConv2d(in_c, out_c, 3, padding=1, kernel_num=2),
            nn.BatchNorm2d(out_c),
            nn.GELU()
        )

    def forward(self, x):
        return self.conv(x)

