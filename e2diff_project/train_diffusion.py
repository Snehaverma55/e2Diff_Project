"""Stage 2 - train the E2Diff denoising generator on HQ faces (Table 1 hyper-parameters)."""
import argparse, os, torch, lpips
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm import tqdm
from e2diff import E2DiffUNet, GaussianDiffusion
from e2diff.data import FaceFolder
from e2diff.utils import EMA, count_params

p = argparse.ArgumentParser()
p.add_argument('--data', required=True, help='folder with HQ face images (FFHQ)')
p.add_argument('--out', default='checkpoints/diffusion')
p.add_argument('--image_size', type=int, default=512)
p.add_argument('--batch_size', type=int, default=2)
p.add_argument('--lr', type=float, default=1e-4)
p.add_argument('--iters', type=int, default=800_000)
p.add_argument('--T', type=int, default=1000)
p.add_argument('--schedule', default='linear')
p.add_argument('--ema', type=float, default=0.999)
p.add_argument('--w_noise', type=float, default=1.0)
p.add_argument('--w_l1', type=float, default=1.0)
p.add_argument('--w_lpips', type=float, default=0.1)
p.add_argument('--aux_t_max', type=int, default=500, help='L1/LPIPS only for t < this (set 0 to use noise loss only)')
p.add_argument('--save_every', type=int, default=10_000)
p.add_argument('--workers', type=int, default=4)
p.add_argument('--resume', default='')
p.add_argument('--mixed_precision', default='no', choices=['no', 'fp16', 'bf16'])
a = p.parse_args()

acc = Accelerator(mixed_precision=a.mixed_precision)
unet = E2DiffUNet(image_size=a.image_size)
diff = GaussianDiffusion(a.T, a.schedule)
opt = torch.optim.AdamW(unet.parameters(), lr=a.lr, betas=(0.9, 0.999))
ema, step = EMA(unet, a.ema), 0
if a.resume:
    ck = torch.load(a.resume, map_location='cpu')
    unet.load_state_dict(ck['model']); ema.shadow.load_state_dict(ck['ema']); opt.load_state_dict(ck['opt']); step = ck['step']
if acc.is_main_process:
    print(f'U-Net parameters: {count_params(unet):.2f} M')
dl = DataLoader(FaceFolder(a.data, a.image_size, 'hq'), batch_size=a.batch_size, shuffle=True,
                num_workers=a.workers, drop_last=True, pin_memory=True)
unet, opt, dl = acc.prepare(unet, opt, dl)
ema.shadow.to(acc.device)
use_aux = a.aux_t_max > 0 and (a.w_l1 > 0 or a.w_lpips > 0)
lp = lpips.LPIPS(net='vgg').to(acc.device).eval().requires_grad_(False) if (use_aux and a.w_lpips > 0) else None
os.makedirs(a.out, exist_ok=True)

bar = tqdm(total=a.iters, initial=step, disable=not acc.is_main_process)
while step < a.iters:
    for x0 in dl:
        t = torch.randint(0, a.T, (x0.shape[0],), device=x0.device)
        loss, logs = diff.training_losses(unet, x0, t, lp, a.w_noise, a.w_l1 if use_aux else 0,
                                          a.w_lpips if use_aux else 0, a.aux_t_max)
        opt.zero_grad(set_to_none=True); acc.backward(loss)
        acc.clip_grad_norm_(unet.parameters(), 1.0); opt.step()
        ema.update(acc.unwrap_model(unet))
        step += 1; bar.update(1); bar.set_postfix({k: f'{v.item():.4f}' for k, v in logs.items()})
        if acc.is_main_process and (step % a.save_every == 0 or step == a.iters):
            ck = dict(model=acc.unwrap_model(unet).state_dict(), ema=ema.shadow.state_dict(), opt=opt.state_dict(), step=step)
            torch.save(ck, f'{a.out}/e2diff_{step}.pt'); torch.save(ck, f'{a.out}/e2diff_latest.pt')
        if step >= a.iters:
            break
