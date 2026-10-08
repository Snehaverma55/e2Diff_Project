"""Building blocks of the E2Diff denoising generator: LRM, self-attention, TES."""
import math, torch, torch.nn as nn, torch.nn.functional as F


def GN(ch):
    return nn.GroupNorm(32, ch)


def timestep_embedding(t, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([args.cos(), args.sin()], dim=-1)
    return F.pad(emb, (0, 1)) if dim % 2 else emb


def zero(m):
    for p in m.parameters():
        nn.init.zeros_(p)
    return m


class LRM(nn.Module):
    """Lightweight Residual Module (paper Eqs. 8-11).
    input layer : GN -> GELU -> depth-wise 3x3 -> point-wise 1x1          (Eq. 8)
    embed layer : GELU -> Linear(time embedding)                          (Eq. 9)
    output layer: GN(F_in + F_emb) -> GELU -> Dropout -> 3x3 conv         (Eq. 10)
    output      : F_in + F_out                                            (Eq. 11)
    """
    def __init__(self, in_ch, out_ch, emb_dim, dropout=0.0):
        super().__init__()
        self.norm_in = GN(in_ch)
        self.dw = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, 1)
        self.emb = nn.Linear(emb_dim, out_ch)
        self.norm_out = GN(out_ch)
        self.drop = nn.Dropout(dropout)
        self.conv_out = zero(nn.Conv2d(out_ch, out_ch, 3, padding=1))

    def forward(self, x, emb):
        f_in = self.pw(self.dw(F.gelu(self.norm_in(x))))
        f_emb = self.emb(F.gelu(emb))[:, :, None, None]
        f_out = self.conv_out(self.drop(F.gelu(self.norm_out(f_in + f_emb))))
        return f_in + f_out


class SelfAttention(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.heads = max(1, ch // 64)
        self.norm = GN(ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = zero(nn.Conv2d(ch, ch, 1))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv(self.norm(x)).reshape(b, 3, self.heads, c // self.heads, h * w)
        q, k, v = [qkv[:, i].transpose(-1, -2) for i in range(3)]       # (b, heads, hw, d)
        o = F.scaled_dot_product_attention(q, k, v)                      # (b, heads, hw, d)
        o = o.transpose(-1, -2).reshape(b, c, h, w)
        return x + self.proj(o)


class TES(nn.Module):
    """Timestep-Embedded Sequential block = LRM (+ self-attention at low resolutions)."""
    def __init__(self, in_ch, out_ch, emb_dim, attn=False, dropout=0.0):
        super().__init__()
        self.lrm = LRM(in_ch, out_ch, emb_dim, dropout)
        self.attn = SelfAttention(out_ch) if attn else nn.Identity()

    def forward(self, x, emb):
        return self.attn(self.lrm(x, emb))


def resample(x, scale):
    """Bicubic up/down-sampling (used instead of strided / transposed convs, per the paper)."""
    dt = x.dtype
    kw = dict(antialias=True) if scale < 1 else {}
    return F.interpolate(x.float(), scale_factor=scale, mode='bicubic', align_corners=False, **kw).to(dt)
