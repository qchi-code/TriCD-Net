import numbers
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from pdb import set_trace as stx

def weight_init(module):
    for n, m in module.named_children():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.LayerNorm)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, (
        nn.ReLU, nn.Sigmoid, nn.Softmax, nn.PReLU, nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d, nn.AdaptiveAvgPool1d,
        nn.Sigmoid, nn.Identity)):
            pass
        else:
            m.initialize()


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

    def initialize(self):
        weight_init(self)


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

    def initialize(self):
        weight_init(self)


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

    def initialize(self):
        weight_init(self)


# ===== ������ǿ��֧�ַ��� per-head ��� =====
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias, mode):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv_0 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)  # Q
        self.qkv_1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)  # K
        self.qkv_2 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)  # V

        self.qkv1conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.qkv2conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.qkv3conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, mask=None, return_head=False):
        b, c, h, w = x.shape
        q = self.qkv1conv(self.qkv_0(x))
        k = self.qkv2conv(self.qkv_1(x))
        v = self.qkv3conv(self.qkv_2(x))
        if mask is not None:
            # mask: (b,1,h,w) or (b,c,h,w); �㲥��
            q = q * mask
            k = k * mask

        # b, (head*c_h), h, w -> b, head, c_h, hw
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature  # (b, head, hw, hw)
        attn = attn.softmax(dim=-1)

        out_h = (attn @ v)  # (b, head, c_h, hw)
        out = rearrange(out_h, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)  # (b, c, h, w)

        if return_head:
            out_h = rearrange(out_h, 'b head c (h w) -> b head c h w', h=h, w=w)
            return out, out_h  # �ϲ������ + per-head ���
        return out

    def initialize(self):
        weight_init(self)


class MSA_head(nn.Module):
    def __init__(self, mode='dilation', dim=256, num_heads=8, ffn_expansion_factor=4, bias=False,
                 LayerNorm_type='WithBias'):
        super(MSA_head, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias, mode)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        self.num_heads = num_heads

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask, return_head=False)
        x = x + self.ffn(self.norm2(x))
        return x

    @torch.no_grad()
    def _shape_check(self, x):
        assert x.dim() == 4, "Expected (B,C,H,W)"

    # ���������� per-head ע����������� FFN ǰ�������ڶ�̬��Ȩ
    def forward_with_heads(self, x, mask=None):
        self._shape_check(x)
        res = x
        attn_out, out_heads = self.attn(self.norm1(x), mask, return_head=True)  # out_heads: (B,head,C_h,H,W)
        x = res + attn_out
        y = self.ffn(self.norm2(x))
        x = x + y  # ��֧�������������в��ſ��ںϣ�
        return x, out_heads  # (B,C,H,W), (B,head,C_h,H,W)

    def initialize(self):
        weight_init(self)


# ===== ������������ע������ǰ��/������ѯ �� �߽��ֵ�� =====
class CrossAttention(nn.Module):
    def __init__(self, dim=256, num_heads=8, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # Q from q_x
        self.q0 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.qdw = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=bias)
        # K,V from kv_x
        self.k0 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.kdw = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=bias)
        self.v0 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.vdw = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, q_x, kv_x, q_mask=None, kv_mask=None, return_head=False):
        b, c, h, w = q_x.shape
        q = self.qdw(self.q0(q_x))
        k = self.kdw(self.k0(kv_x))
        v = self.vdw(self.v0(kv_x))
        if q_mask is not None:
            q = q * q_mask
        if kv_mask is not None:
            k = k * kv_mask

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out_h = (attn @ v)  # (b, head, c_h, hw)
        out = rearrange(out_h, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        if return_head:
            out_h = rearrange(out_h, 'b head c (h w) -> b head c h w', h=h, w=w)
            return out, out_h
        return out

    def initialize(self):
        weight_init(self)



class HeadAllocator(nn.Module):

    def __init__(self, dim=256, num_heads=8, hidden=None, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.eps = eps
        hidden = hidden or dim  # ���� MLP
        self.mlp = nn.Sequential(
            nn.Linear(3 * dim + 3, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_heads * 3)
        )

    def masked_mean(self, F, M):  # F: (B,C,H,W), M: (B,1,H,W)
        num = (F * M).sum(dim=(2, 3))  # (B,C)
        den = M.sum(dim=(2, 3)) + self.eps  # (B,1)
        return num / den  # (B,C)

    def forward(self, F, masks: dict):
        B, C, H, W = F.shape
        M_fg, M_bg, M_bd = masks['fg'], masks['bg'], masks['bd']  # (B,1,H,W)

        mu_fg = self.masked_mean(F, M_fg)  # (B,C)
        mu_bg = self.masked_mean(F, M_bg)
        mu_bd = self.masked_mean(F, M_bd)

        s_fg = M_fg.sum(dim=(2, 3), keepdim=False) / float(H * W)  # (B,1)
        s_bg = M_bg.sum(dim=(2, 3), keepdim=False) / float(H * W)
        s_bd = M_bd.sum(dim=(2, 3), keepdim=False) / float(H * W)

        d = torch.cat([mu_fg, mu_bg, mu_bd, s_fg, s_bg, s_bd], dim=1)  # (B, 3C+3)
        logits = self.mlp(d).view(B, self.num_heads, 3)  # (B, heads, 3)
        g = torch.softmax(logits, dim=-1)  # ÿ�� head �� {fg,bg,bd} ��Ȩ��
        return g  # (B, heads, 3)

    def initialize(self):
        weight_init(self)


# ===== ������FEBR ģ�飨����֧ + CRA + ��̬ͷ��Ȩ + �в��ſ��ںϣ� =====
class FEBR(nn.Module):
    def __init__(self, dim=256, num_heads=8, bias=False, LayerNorm_type='WithBias'):
        super().__init__()
        # ����֧ MSA����ԭʵ�ֱ���һ�£�
        self.F_TA = MSA_head(dim=dim, num_heads=num_heads, bias=bias, LayerNorm_type=LayerNorm_type)
        self.B_TA = MSA_head(dim=dim, num_heads=num_heads, bias=bias, LayerNorm_type=LayerNorm_type)
        self.BD_TA = MSA_head(dim=dim, num_heads=num_heads, bias=bias, LayerNorm_type=LayerNorm_type)

        # ����������ע����
        self.cra_fg_bd = CrossAttention(dim=dim, num_heads=num_heads, bias=bias)  # fg <- bd
        self.cra_bg_bd = CrossAttention(dim=dim, num_heads=num_heads, bias=bias)  # bg <- bd

        # ��̬ͷ����
        self.allocator = HeadAllocator(dim=dim, num_heads=num_heads)

        # ͷ�ϲ���в��ſ�
        self.head_fuse = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.gate_conv = nn.Conv2d(3 * dim, dim, kernel_size=3, padding=1, bias=bias)

        # CRA Ȩ�أ���ѧϰ������
        self.beta_fg = nn.Parameter(torch.tensor(1.0))
        self.beta_bg = nn.Parameter(torch.tensor(1.0))

        self.num_heads = num_heads
        self.dim = dim

    def forward(self, qry_fts, fg_prob, bd_mask):

        B, C, H, W = qry_fts.shape
        fg_mask = fg_prob
        bg_mask = 1.0 - fg_mask

        # ����֧�����ط�֧�������� + per-head ע������������ڶ�̬��Ȩ��
        xf, Xh_fg = self.F_TA.forward_with_heads(qry_fts, fg_mask)   # (B,C,H,W), (B,head,C_h,H,W)
        xb, Xh_bg = self.B_TA.forward_with_heads(qry_fts, bg_mask)
        xbd, Xh_bd = self.BD_TA.forward_with_heads(qry_fts, bd_mask)

        # ������ע������ֻ��Ҫ per-head ���������ϣ�
        _, Ch_fg_bd = self.cra_fg_bd(qry_fts, qry_fts, q_mask=fg_mask, kv_mask=bd_mask, return_head=True)  # fg <- bd
        _, Ch_bg_bd = self.cra_bg_bd(qry_fts, qry_fts, q_mask=bg_mask, kv_mask=bd_mask, return_head=True)  # bg <- bd

        # ��̬ͷ���䣨��·�ɣ�
        g = self.allocator(qry_fts, {'fg': fg_mask, 'bg': bg_mask, 'bd': bd_mask})  # (B,head,3)
        g_fg = g[:, :, 0].view(B, self.num_heads, 1, 1, 1)
        g_bg = g[:, :, 1].view(B, self.num_heads, 1, 1, 1)
        g_bd = g[:, :, 2].view(B, self.num_heads, 1, 1, 1)

        # ��ϣ�Y_t = ��_r g_t^r X_t^r + g_t^fg ��_fg C_t^{fg<-bd} + g_t^bg ��_bg C_t^{bg<-bd}
        Yh = g_fg * Xh_fg + g_bg * Xh_bg + g_bd * Xh_bd \
             + g_fg * self.beta_fg * Ch_fg_bd + g_bg * self.beta_bg * Ch_bg_bd  # (B,head,C_h,H,W)
        Y = rearrange(Yh, 'b head c h w -> b (head c) h w')  # (B,C,H,W)
        F_tilde = self.head_fuse(Y)  # ����ͶӰ�ںϸ� head

        # �в��ſ��ںϣ�����ƴ�ӻ��ڷ�֧������������ FFN �ģ�
        gamma = torch.sigmoid(self.gate_conv(torch.cat([xf, xb, xbd], dim=1)))  # (B,C,H,W)
        F_star = gamma * F_tilde + (1.0 - gamma) * qry_fts
        return F_star

    def initialize(self):
        weight_init(self)