"""Create a synthetic CelebA-Test-style set: python make_lq.py --hq celeba_hq/ --out celeba_test --n 4000"""
import argparse, os, cv2, random
from e2diff.data import list_images
from e2diff.degradation import random_degrade

p = argparse.ArgumentParser()
p.add_argument('--hq', required=True); p.add_argument('--out', default='celeba_test')
p.add_argument('--n', type=int, default=4000); p.add_argument('--size', type=int, default=512)
p.add_argument('--seed', type=int, default=0)
a = p.parse_args()
random.seed(a.seed)
os.makedirs(f'{a.out}/lq', exist_ok=True); os.makedirs(f'{a.out}/hq', exist_ok=True)
for f in list_images(a.hq)[:a.n]:
    n = os.path.splitext(os.path.basename(f))[0]
    hq = cv2.resize(cv2.imread(f), (a.size, a.size), interpolation=cv2.INTER_AREA)
    cv2.imwrite(f'{a.out}/hq/{n}.png', hq); cv2.imwrite(f'{a.out}/lq/{n}.png', random_degrade(hq))
