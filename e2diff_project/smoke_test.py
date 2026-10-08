"""Quick CPU sanity check with a tiny model: python smoke_test.py"""
import torch
from e2diff import E2DiffUNet, GaussianDiffusion
from e2diff.dpm_solver import dpm_solver_sample

torch.manual_seed(0)
net = E2DiffUNet(base=32, mults=(1, 2), num_blocks=(1, 1), attn_res=(16,), image_size=32).eval()
diff = GaussianDiffusion(1000)
x0 = torch.randn(2, 3, 32, 32).clamp(-1, 1)
t = torch.randint(0, 1000, (2,))
loss, logs = diff.training_losses(net, x0, t, None, 1.0, 1.0, 0.0, 500)
loss.backward(); print('train loss ok', float(loss))
xn = diff.q_sample(x0, torch.full((2,), 399), torch.randn_like(x0))
for steps in (1, 10, 25):
    out = dpm_solver_sample(net, xn, diff.alphas_cumprod, 400, steps=steps)
    assert out.shape == x0.shape and torch.isfinite(out).all()
print('dpm-solver ok'); print('ddpm ok', diff.ddpm_sample(net, xn, 5).shape)
