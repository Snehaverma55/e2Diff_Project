"""Full E2Diff inference: predictor -> HFGM -> one-step diffusion to N -> DPM-Solver denoising (Fig. 1)."""
import torch
from .dpm_solver import dpm_solver_sample


class E2DiffPipeline:
    def __init__(self, unet, diffusion, predictor, hfgm=None, N=400, steps=50, sampler='dpm',
                 order=2, device='cuda', amp_dtype=None):
        self.unet, self.diff, self.pred, self.hfgm = unet, diffusion, predictor, hfgm
        self.N, self.steps, self.sampler, self.order, self.device, self.amp = N, steps, sampler, order, device, amp_dtype

    @torch.no_grad()
    def restore(self, lq, generator=None):
        """lq: Bx3x512x512 in [-1,1] (already bicubic-resized to 512). Returns restored tensor in [-1,1]."""
        lq = lq.to(self.device)
        x_i = self.pred(lq)                                      # low-frequency estimate
        if self.hfgm is not None:
            x_i = self.hfgm(x_i)                                 # + high-frequency details
        n = self.N - 1                                           # index of step N
        t = torch.full((x_i.shape[0],), n, device=self.device, dtype=torch.long)
        noise = torch.randn(x_i.shape, device=self.device, generator=generator)
        x_n = self.diff.q_sample(x_i, t, noise)                  # one-step diffusion, Eq. (5)
        if self.sampler == 'dpm':
            x0 = dpm_solver_sample(self.unet, x_n, self.diff.alphas_cumprod, self.N,
                                   steps=self.steps, order=self.order, autocast=self.amp)
        else:
            x0 = self.diff.ddpm_sample(self.unet, x_n, self.N)
        return x0.clamp(-1, 1)
