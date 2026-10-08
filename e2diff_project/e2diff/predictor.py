"""Diffusion predictor f(y0; w) that produces the initial estimate x_I from the LQ image.
The paper (like DifFace) uses SwinIR. Offered here:
  * 'rrdb'   : self-contained RRDB restoration net (no extra dependency, trainable with train_predictor.py)
  * 'swinir' : official SwinIR via BasicSR (load DifFace's pretrained restoration weights)."""
import torch, torch.nn as nn, torch.nn.functional as F


class RDB(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.c = nn.ModuleList([nn.Conv2d(nf + i * gc, gc if i < 4 else nf, 3, padding=1) for i in range(5)])

    def forward(self, x):
        feats = [x]
        for i, c in enumerate(self.c):
            o = c(torch.cat(feats, 1))
            if i < 4:
                feats.append(F.leaky_relu(o, 0.2))
        return o * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.b = nn.Sequential(RDB(nf, gc), RDB(nf, gc), RDB(nf, gc))

    def forward(self, x):
        return self.b(x) * 0.2 + x


class RRDBRestorer(nn.Module):
    def __init__(self, nf=64, nb=8, gc=32):
        super().__init__()
        self.first = nn.Conv2d(3, nf, 3, padding=1)
        self.trunk = nn.Sequential(*[RRDB(nf, gc) for _ in range(nb)])
        self.trunk_conv = nn.Conv2d(nf, nf, 3, padding=1)
        self.hr = nn.Conv2d(nf, nf, 3, padding=1)
        self.last = nn.Conv2d(nf, 3, 3, padding=1)

    def forward(self, x):                         # x in [-1,1]
        f = self.first(x)
        f = f + self.trunk_conv(self.trunk(f))
        return (x + self.last(F.leaky_relu(self.hr(f), 0.2))).clamp(-1, 1)


class SwinIRWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        from basicsr.archs.swinir_arch import SwinIR      # pip install basicsr
        self.net = SwinIR(upscale=1, in_chans=3, img_size=64, window_size=8, img_range=1.,
                          depths=[6] * 6, embed_dim=180, num_heads=[6] * 6, mlp_ratio=2,
                          upsampler='', resi_connection='1conv')

    def forward(self, x):                         # SwinIR works in [0,1]
        return (self.net((x + 1) / 2) * 2 - 1).clamp(-1, 1)


def build_predictor(name='rrdb'):
    return RRDBRestorer() if name == 'rrdb' else SwinIRWrapper()


def load_predictor(name, ckpt, device):
    m = build_predictor(name)
    if ckpt:
        sd = torch.load(ckpt, map_location='cpu')
        sd = sd.get('params_ema', sd.get('params', sd.get('model', sd)))
        target = m.net if name == 'swinir' else m
        target.load_state_dict(sd, strict=True)
    return m.to(device).eval().requires_grad_(False)
