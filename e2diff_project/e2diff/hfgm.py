"""High-Frequency Feature Generation Module (paper Sec. III-D, Fig. 3).
Plug-and-play: the pretrained CodeFormer (encoder -> Transformer code predictor -> codebook -> decoder)
maps the low-frequency estimate to its high-frequency counterpart. No extra training is needed.
Setup:  git clone https://github.com/sczhou/CodeFormer ; download weights/CodeFormer/codeformer.pth"""
import sys, torch, torch.nn as nn


class HFGM(nn.Module):
    def __init__(self, codeformer_repo, ckpt, fidelity=0.5):
        super().__init__()
        sys.path.insert(0, codeformer_repo)
        from basicsr.archs.codeformer_arch import CodeFormer     # shipped inside the CodeFormer repo
        self.net = CodeFormer(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9,
                              connect_list=['32', '64', '128', '256'])
        sd = torch.load(ckpt, map_location='cpu')
        self.net.load_state_dict(sd.get('params_ema', sd))
        self.w = fidelity

    @torch.no_grad()
    def forward(self, x):                           # x: Bx3x512x512 in [-1,1] (RGB)
        return self.net(x, w=self.w, adain=True)[0].clamp(-1, 1)
