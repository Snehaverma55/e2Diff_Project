"""PSNR / SSIM / LPIPS / LMD between restored images and ground truth (matched by file name).
FID-G / FID-F:  python -m pytorch_fid <restored_dir> <gt_dir>   |   python -m pytorch_fid <restored_dir> <ffhq_dir>"""
import argparse, os, cv2, numpy as np, torch, lpips
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from e2diff.utils import bgr_to_tensor

p = argparse.ArgumentParser()
p.add_argument('--restored', required=True); p.add_argument('--gt', required=True)
p.add_argument('--lmd', action='store_true', help='also compute landmark distance (needs face-alignment)')
a = p.parse_args()
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
lp = lpips.LPIPS(net='vgg').to(dev).eval()
fa = None
if a.lmd:
    import face_alignment
    lt = getattr(face_alignment.LandmarksType, 'TWO_D', None) or face_alignment.LandmarksType._2D
    fa = face_alignment.FaceAlignment(lt, device=dev)

R = {'psnr': [], 'ssim': [], 'lpips': [], 'lmd': []}
for f in sorted(os.listdir(a.restored)):
    g = next((os.path.join(a.gt, f[:-len(e)] + e2) for e in [os.path.splitext(f)[1]]
              for e2 in ['.png', '.jpg', '.jpeg'] if os.path.exists(os.path.join(a.gt, f[:-len(e)] + e2))), None)
    if g is None:
        continue
    x, y = cv2.imread(os.path.join(a.restored, f)), cv2.imread(g)
    y = cv2.resize(y, (x.shape[1], x.shape[0]))
    R['psnr'].append(psnr(y, x)); R['ssim'].append(ssim(y, x, channel_axis=2))
    with torch.no_grad():
        R['lpips'].append(lp(bgr_to_tensor(x)[None].to(dev), bgr_to_tensor(y)[None].to(dev)).item())
    if fa is not None:
        lx, ly = fa.get_landmarks(x[..., ::-1].copy()), fa.get_landmarks(y[..., ::-1].copy())
        if lx and ly:
            R['lmd'].append(np.linalg.norm(lx[0] - ly[0], axis=1).mean())
print(f"images: {len(R['psnr'])}")
print(f"PSNR {np.mean(R['psnr']):.2f}  SSIM {np.mean(R['ssim']):.3f}  LPIPS {np.mean(R['lpips']):.3f}" +
      (f"  LMD {np.mean(R['lmd']):.2f}" if R['lmd'] else ''))
