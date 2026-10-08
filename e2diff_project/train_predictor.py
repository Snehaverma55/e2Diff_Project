"""Stage 1 - train the diffusion predictor f(y0; w) (LQ -> low-frequency estimate) with L1 + LPIPS."""
import argparse, os, torch, lpips
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm import tqdm
from e2diff.data import FaceFolder
from e2diff.predictor import build_predictor

p = argparse.ArgumentParser()
p.add_argument('--data', required=True, help='folder with HQ face images (FFHQ)')
p.add_argument('--out', default='checkpoints/predictor')
p.add_argument('--arch', default='rrdb', choices=['rrdb'])
p.add_argument('--image_size', type=int, default=512)
p.add_argument('--crop', type=int, default=256, help='random crop of the (LQ,HQ) pair to save memory')
p.add_argument('--batch_size', type=int, default=4)
p.add_argument('--lr', type=float, default=2e-4)
p.add_argument('--iters', type=int, default=200_000)
p.add_argument('--w_lpips', type=float, default=0.1)
p.add_argument('--save_every', type=int, default=10_000)
p.add_argument('--workers', type=int, default=4)
p.add_argument('--mixed_precision', default='no', choices=['no', 'fp16', 'bf16'])
a = p.parse_args()

acc = Accelerator(mixed_precision=a.mixed_precision)
net = build_predictor(a.arch)
opt = torch.optim.AdamW(net.parameters(), lr=a.lr, betas=(0.9, 0.999))
dl = DataLoader(FaceFolder(a.data, a.image_size, 'pair', a.crop), batch_size=a.batch_size, shuffle=True,
                num_workers=a.workers, drop_last=True, pin_memory=True)
net, opt, dl = acc.prepare(net, opt, dl)
lp = lpips.LPIPS(net='vgg').to(acc.device).eval().requires_grad_(False)
os.makedirs(a.out, exist_ok=True)

step, bar = 0, tqdm(total=a.iters, disable=not acc.is_main_process)
while step < a.iters:
    for lq, hq in dl:
        out = net(lq)
        l1 = (out - hq).abs().mean()
        loss = l1 + a.w_lpips * lp(out.float(), hq.float()).mean()
        opt.zero_grad(set_to_none=True); acc.backward(loss); opt.step()
        step += 1; bar.update(1); bar.set_postfix(l1=f'{l1.item():.4f}', loss=f'{loss.item():.4f}')
        if acc.is_main_process and (step % a.save_every == 0 or step == a.iters):
            torch.save(acc.unwrap_model(net).state_dict(), f'{a.out}/predictor_{step}.pt')
            torch.save(acc.unwrap_model(net).state_dict(), f'{a.out}/predictor_latest.pt')
        if step >= a.iters:
            break
