# E2Diff – unofficial PyTorch implementation

Implements *"E2Diff: An Efficient Enhanced Diffusion Model for Blind Face Restoration"* (IEEE Access 2026).
The authors' code is at https://github.com/windcr/E2Diff — use it as the reference if results differ.

## Pipeline (paper Fig. 1)
`LQ -> predictor f (SwinIR/RRDB) -> HFGM (CodeFormer) -> diffuse to step N -> E2Diff U-Net + DPM-Solver -> HQ`

| Paper component | File |
|---|---|
| LRM, TES, attention (Eqs. 8-11, Fig. 2) | `e2diff/modules.py` |
| Denoising generator (U-Net, base 32, mults [1,2,4,8,8,16,16], blocks [1,2,2,2,2,3,4]) | `e2diff/unet.py` |
| Diffusion process, losses (Eqs. 1-7, Table 1) | `e2diff/diffusion.py` |
| DPM-Solver (Eqs. 12-15) | `e2diff/dpm_solver.py` |
| HFGM (CodeFormer, plug-and-play) | `e2diff/hfgm.py` |
| Diffusion predictor | `e2diff/predictor.py` |
| Degradation model (Eq. 16) | `e2diff/degradation.py` |

## 1. Install
```bash
python -m venv venv && source venv/bin/activate        # Windows: venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   # pick your CUDA
pip install -r requirements.txt
python smoke_test.py          # tiny CPU check: training loss + DPM-Solver + DDPM run
python -m e2diff.unet         # prints U-Net parameter count
```

## 2. Data
* Train: **FFHQ** (70k faces) resized to 512x512 -> any folder, e.g. `data/ffhq512/`.
* Test: build a CelebA-Test-style set from CelebA-HQ:
  `python make_lq.py --hq data/celeba_hq --out data/celeba_test --n 4000`
  Real-world sets (LFW-Test, WebPhoto-Test, Wider-Test, CelebChild-Test) are available from the DifFace / CodeFormer repos.

## 3. Training (two stages, as in DifFace)
**Stage 1 – predictor** (LQ -> smooth estimate, L1 + LPIPS):
```bash
accelerate launch train_predictor.py --data data/ffhq512 --out checkpoints/predictor --iters 200000
```
(Alternative: use DifFace's pretrained SwinIR restoration weights and pass `--predictor swinir` at inference; needs `pip install basicsr`.)

**Stage 2 – E2Diff denoiser** (800k iters, AdamW 1e-4, batch 2, EMA 0.999, T=1000, linear schedule):
```bash
accelerate config          # once: choose multi-GPU / bf16 as you like
accelerate launch train_diffusion.py --data data/ffhq512 --out checkpoints/diffusion \
    --mixed_precision bf16
```
Resume with `--resume checkpoints/diffusion/e2diff_latest.pt`. Paper reports ~4 weeks on 4x RTX 3080 Ti.
For a quick experiment: `--image_size 256 --iters 100000` (then skip HFGM, CodeFormer is 512-only).

## 4. HFGM setup (CodeFormer, no training)
```bash
mkdir -p third_party && git clone https://github.com/sczhou/CodeFormer third_party/CodeFormer
# download codeformer.pth from the CodeFormer release page into
# third_party/CodeFormer/weights/CodeFormer/codeformer.pth
pip install -r third_party/CodeFormer/requirements.txt   # (only if imports complain)
```

## 5. Inference
```bash
python infer.py --input data/celeba_test/lq --output results/e2diff \
  --unet_ckpt checkpoints/diffusion/e2diff_latest.pt \
  --predictor rrdb --predictor_ckpt checkpoints/predictor/predictor_latest.pt \
  --N 400 --steps 50 --bf16
```
* `--steps 10|15|20|25|50|100` – DPM-Solver steps (paper Tables 5/6). `--sampler ddpm` = slow N-step baseline.
* `--no_hfgm` – ablation without the high-frequency module.

## 6. Evaluation
```bash
python evaluate.py --restored results/e2diff --gt data/celeba_test/hq --lmd      # PSNR/SSIM/LPIPS/LMD
python -m pytorch_fid results/e2diff data/celeba_test/hq                          # FID-G
python -m pytorch_fid results/e2diff data/ffhq512                                 # FID-F
```
Ablations (Table 4): "Baseline" = standard residual blocks (replace `LRM` by a ResBlock), "W/LRM" = `--no_hfgm`, "W/HFGM" = baseline + HFGM.

## Deviations / things the paper leaves unspecified (please read)
1. **Not tested on GPU here** – only syntax-checked; run `smoke_test.py` first and report tracebacks.
2. **Parameter count will differ** from the paper's 95.91M; the exact channel layout isn't fully specified. Check with `python -m e2diff.unet`.
3. **Embedding layer (Eq. 9)** is applied to the *timestep embedding* (as Fig. 2's "Time Embed" implies), not to `F_LRM(i-1)`.
4. **Learned sigma** (Table 1) is not implemented: the network predicts only the noise; DPM-Solver doesn't use the variance.
5. **L1 / LPIPS losses** (Table 1) are applied to the x0-estimate only for `t < --aux_t_max` (default 500), since estimates at high noise are meaningless. `--aux_t_max 0` gives plain DDPM training.
6. **Start step N** isn't given in the paper; default 400 (DifFace's value). Tune `--N` (larger = more generation, less fidelity).
7. **Predictor**: paper uses SwinIR; a self-contained RRDB net is the default so the repo runs without extra dependencies.
8. Input size is fixed to 512x512 (as stated in the paper's Discussion).
9. DPM-Solver is singlestep order 2 (order-1 final step for odd step counts), time-uniform grid.
