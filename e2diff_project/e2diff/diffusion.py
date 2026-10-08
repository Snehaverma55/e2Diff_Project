"""Gaussian diffusion (paper Eqs. 1-7): forward process, training loss, ancestral sampler."""
import math, numpy as np, torch, torch.nn.functional as F


def make_betas(T, schedule='linear'):
    if schedule == 'linear':
        s = 1000 / T
        return np.linspace(s * 1e-4, s * 0.02, T, dtype=np.float64)
    if schedule == 'cosine':
        f = lambda t: math.cos((t / T + 0.008) / 1.008 * math.pi / 2) ** 2
        return np.array([min(1 - f(i + 1) / f(i), 0.999) for i in range(T)], dtype=np.float64)
    raise ValueError(schedule)


def _ex(arr, t, shape):
    out = arr.to(t.device)[t].float()
    return out.view(-1, *([1] * (len(shape) - 1)))


class GaussianDiffusion:
    def __init__(self, T=1000, schedule='linear'):
        self.T = T
        b = make_betas(T, schedule)
        a = 1 - b
        ac = np.cumprod(a)
        ac_prev = np.append(1.0, ac[:-1])
        tt = lambda x: torch.tensor(x, dtype=torch.float64)
        self.betas, self.alphas, self.alphas_cumprod = tt(b), tt(a), tt(ac)
        self.sqrt_ac, self.sqrt_1m_ac = tt(np.sqrt(ac)), tt(np.sqrt(1 - ac))
        self.post_var = tt(b * (1 - ac_prev) / (1 - ac))          # sigma_t^2 in Eq. (4)
        self.sqrt_recip_a = tt(1 / np.sqrt(a))

    # ---- forward process, Eq. (3)
    def q_sample(self, x0, t, noise=None):
        noise = torch.randn_like(x0) if noise is None else noise
        return _ex(self.sqrt_ac, t, x0.shape) * x0 + _ex(self.sqrt_1m_ac, t, x0.shape) * noise

    def pred_x0(self, xt, t, eps):
        return (xt - _ex(self.sqrt_1m_ac, t, xt.shape) * eps) / _ex(self.sqrt_ac, t, xt.shape)

    # ---- training: noise-prediction loss + (optional) L1 / LPIPS on the x0 estimate (Table 1 weights)
    def training_losses(self, model, x0, t, lpips_fn=None, w_noise=1.0, w_l1=1.0, w_lpips=0.1, aux_t_max=500):
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        eps = model(xt, t)
        l_noise = F.mse_loss(eps.float(), noise.float())
        total, logs = w_noise * l_noise, {'noise': l_noise.detach()}
        m = t < aux_t_max            # x0 estimates are only meaningful at low noise levels
        if m.any() and (w_l1 > 0 or (w_lpips > 0 and lpips_fn is not None)):
            x0p = self.pred_x0(xt[m].float(), t[m], eps[m].float()).clamp(-1, 1)
            if w_l1 > 0:
                l1 = F.l1_loss(x0p, x0[m].float()); total = total + w_l1 * l1; logs['l1'] = l1.detach()
            if w_lpips > 0 and lpips_fn is not None:
                lp = lpips_fn(x0p, x0[m].float()).mean(); total = total + w_lpips * lp; logs['lpips'] = lp.detach()
        return total, logs

    # ---- reverse process, Eq. (4): ancestral sampling from step N (baseline, slow)
    @torch.no_grad()
    def ddpm_sample(self, model, x, N, progress=False):
        rng = range(N - 1, -1, -1)
        if progress:
            from tqdm import tqdm; rng = tqdm(rng)
        for i in rng:
            t = torch.full((x.shape[0],), i, device=x.device, dtype=torch.long)
            eps = model(x, t).float()
            mean = _ex(self.sqrt_recip_a, t, x.shape) * (x - _ex(self.betas, t, x.shape) /
                                                         _ex(self.sqrt_1m_ac, t, x.shape) * eps)
            if i > 0:
                x = mean + _ex(self.post_var, t, x.shape).sqrt() * torch.randn_like(x)
            else:
                x = mean
        return x
