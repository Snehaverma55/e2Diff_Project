"""E2Diff denoising generator: U-Net built from TES modules (paper Sec. III-C, IV-B)."""
import torch, torch.nn as nn, torch.nn.functional as F
from .modules import TES, GN, timestep_embedding, resample, zero


class E2DiffUNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, base=32, mults=(1, 2, 4, 8, 8, 16, 16),
                 num_blocks=(1, 2, 2, 2, 2, 3, 4), attn_res=(32, 16, 8), image_size=512, dropout=0.0):
        super().__init__()
        assert len(mults) == len(num_blocks)
        self.base, L = base, len(mults)
        emb_dim = base * 4
        self.time_mlp = nn.Sequential(nn.Linear(base, emb_dim), nn.GELU(), nn.Linear(emb_dim, emb_dim))
        self.conv_in = nn.Conv2d(in_ch, base, 3, padding=1)

        self.enc, ch = nn.ModuleList(), base
        for i in range(L):
            res, out = image_size >> i, base * mults[i]
            blocks = nn.ModuleList()
            for _ in range(num_blocks[i]):
                blocks.append(TES(ch, out, emb_dim, res in attn_res, dropout)); ch = out
            self.enc.append(blocks)

        mid_attn = (image_size >> (L - 1)) in attn_res
        self.mid = nn.ModuleList([TES(ch, ch, emb_dim, mid_attn, dropout), TES(ch, ch, emb_dim, False, dropout)])

        self.dec = nn.ModuleList()                       # deepest level first
        for i in reversed(range(L)):
            res, out = image_size >> i, base * mults[i]
            blocks = nn.ModuleList()
            for _ in range(num_blocks[i]):
                blocks.append(TES(ch + out, out, emb_dim, res in attn_res, dropout)); ch = out
            self.dec.append(blocks)

        self.out = nn.Sequential(GN(ch), nn.GELU(), zero(nn.Conv2d(ch, out_ch, 3, padding=1)))

    def forward(self, x, t):
        L = len(self.enc)
        emb = self.time_mlp(timestep_embedding(t, self.base))
        h, skips = self.conv_in(x), []
        for i, blocks in enumerate(self.enc):
            for b in blocks:
                h = b(h, emb); skips.append(h)
            if i < L - 1:
                h = resample(h, 0.5)
        for b in self.mid:
            h = b(h, emb)
        for k, blocks in enumerate(self.dec):
            for b in blocks:
                h = b(torch.cat([h, skips.pop()], 1), emb)
            if L - 1 - k > 0:
                h = resample(h, 2.0)
        return self.out(h)


if __name__ == '__main__':      # python -m e2diff.unet
    m = E2DiffUNet()
    print(f'E2Diff U-Net parameters: {sum(p.numel() for p in m.parameters())/1e6:.2f} M')
