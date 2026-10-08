"""DPM-Solver (Lu et al. 2022; paper Eqs. 12-15), noise-prediction, singlestep orders 1 & 2.
Re-implemented for the discrete-time VP schedule; starts from an arbitrary t_start (= N/T) because
E2Diff only diffuses the enhanced estimate x_I to step N, not to T."""
import torch


def interp1d(x, xp, fp):
    idx = torch.searchsorted(xp, x.contiguous()).clamp(1, len(xp) - 1)
    x0, x1, y0, y1 = xp[idx - 1], xp[idx], fp[idx - 1], fp[idx]
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


class NoiseSchedule:
    def __init__(self, alphas_cumprod, device):
        self.T = len(alphas_cumprod)
        self.t_arr = torch.linspace(0., 1., self.T + 1, dtype=torch.float64)[1:].to(device)   # t_n = (n+1)/T
        self.log_alpha_arr = (0.5 * torch.log(alphas_cumprod.double())).to(device)

    def log_alpha(self, t):  return interp1d(t, self.t_arr, self.log_alpha_arr)
    def sigma(self, t):      return torch.sqrt(1. - torch.exp(2. * self.log_alpha(t)))
    def lam(self, t):        return self.log_alpha(t) - torch.log(self.sigma(t))

    def inv_lambda(self, lam):
        la = -0.5 * torch.logaddexp(torch.zeros_like(lam), -2. * lam)
        return interp1d(la, torch.flip(self.log_alpha_arr, [0]), torch.flip(self.t_arr, [0]))


@torch.no_grad()
def dpm_solver_sample(model, x, alphas_cumprod, N, steps=50, order=2, skip='time_uniform', autocast=None):
    """model(x, t_index_float) -> eps.  N: starting step (x is x_N).  steps: number of model evaluations."""
    dev, T = x.device, len(alphas_cumprod)
    ns = NoiseSchedule(alphas_cumprod, dev)
    t_start, t_end = torch.tensor([N / T], dtype=torch.float64, device=dev), torch.tensor([1. / T], dtype=torch.float64, device=dev)

    def eps_fn(x_, t_):
        ti = ((t_ - 1. / T) * T).float().expand(x_.shape[0])
        with torch.autocast(device_type=dev.type, dtype=autocast, enabled=autocast is not None):
            return model(x_, ti).float()

    orders = [2] * (steps // 2) + ([1] if steps % 2 else []) if order == 2 else [1] * steps
    K = len(orders)
    if skip == 'time_uniform':
        ts = torch.linspace(t_start.item(), t_end.item(), K + 1, dtype=torch.float64, device=dev)
    else:  # logSNR
        lam = torch.linspace(ns.lam(t_start).item(), ns.lam(t_end).item(), K + 1, dtype=torch.float64, device=dev)
        ts = ns.inv_lambda(lam)

    for i, od in enumerate(orders):
        s, t = ts[i:i + 1], ts[i + 1:i + 2]
        la_s, la_t = ns.log_alpha(s), ns.log_alpha(t)
        sg_s, sg_t = ns.sigma(s), ns.sigma(t)
        h = ns.lam(t) - ns.lam(s)
        e_s = eps_fn(x, s)
        a_ratio = torch.exp(la_t - la_s).float()
        phi1 = torch.expm1(h).float()
        if od == 1:                                           # Eq. (14) with eps held constant
            x = a_ratio * x - sg_t.float() * phi1 * e_s
        else:                                                 # second order, midpoint (r1 = 1/2), Eq. (15)
            r1 = 0.5
            s1 = ns.inv_lambda(ns.lam(s) + r1 * h)
            la_s1, sg_s1 = ns.log_alpha(s1), ns.sigma(s1)
            x_s1 = torch.exp(la_s1 - la_s).float() * x - sg_s1.float() * torch.expm1(r1 * h).float() * e_s
            e_s1 = eps_fn(x_s1, s1)
            x = a_ratio * x - sg_t.float() * phi1 * e_s - (0.5 / r1) * sg_t.float() * phi1 * (e_s1 - e_s)
    return x
