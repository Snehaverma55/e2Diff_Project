"""Restore faces with E2Diff (predictor -> HFGM -> diffusion to N -> DPM-Solver)."""
import argparse, os, cv2, torch
from tqdm import tqdm
from e2diff import E2DiffUNet, GaussianDiffusion, E2DiffPipeline
from e2diff.data import list_images
from e2diff.predictor import load_predictor
from e2diff.utils import bgr_to_tensor, tensor_to_bgr

p = argparse.ArgumentParser()
p.add_argument('--input', required=True, help='image or folder of LQ faces')
p.add_argument('--output', default='results')
p.add_argument('--unet_ckpt', required=True)
p.add_argument('--predictor', default='rrdb', choices=['rrdb', 'swinir'])
p.add_argument('--predictor_ckpt', required=True)
p.add_argument('--codeformer_repo', default='third_party/CodeFormer')
p.add_argument('--codeformer_ckpt', default='third_party/CodeFormer/weights/CodeFormer/codeformer.pth')
p.add_argument('--no_hfgm', action='store_true', help='disable HFGM (ablation "Baseline"/"w/LRM")')
p.add_argument('--fidelity', type=float, default=0.5, help='CodeFormer w inside HFGM')
p.add_argument('--N', type=int, default=400, help='diffusion start step')
p.add_argument('--steps', type=int, default=50, help='sampling steps (paper: 50 or 100; 10-25 for speed)')
p.add_argument('--sampler', default='dpm', choices=['dpm', 'ddpm'])
p.add_argument('--image_size', type=int, default=512)
p.add_argument('--bf16', action='store_true')
p.add_argument('--seed', type=int, default=0)
a = p.parse_args()

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
ck = torch.load(a.unet_ckpt, map_location='cpu')
unet = E2DiffUNet(image_size=a.image_size); unet.load_state_dict(ck['ema'] if 'ema' in ck else ck)
unet = unet.to(dev).eval()
hfgm = None
if not a.no_hfgm:
    from e2diff.hfgm import HFGM
    hfgm = HFGM(a.codeformer_repo, a.codeformer_ckpt, a.fidelity).to(dev).eval()
pipe = E2DiffPipeline(unet, GaussianDiffusion(1000, 'linear'), load_predictor(a.predictor, a.predictor_ckpt, dev),
                      hfgm, a.N, a.steps, a.sampler, device=dev, amp_dtype=torch.bfloat16 if a.bf16 else None)

files = list_images(a.input) if os.path.isdir(a.input) else [a.input]
os.makedirs(a.output, exist_ok=True)
g = torch.Generator(device=dev).manual_seed(a.seed)
for f in tqdm(files):
    img = cv2.resize(cv2.imread(f), (a.image_size, a.image_size), interpolation=cv2.INTER_CUBIC)
    out = pipe.restore(bgr_to_tensor(img)[None], g)
    cv2.imwrite(os.path.join(a.output, os.path.splitext(os.path.basename(f))[0] + '.png'), tensor_to_bgr(out))
