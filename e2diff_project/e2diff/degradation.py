"""Synthetic LQ generation, Eq. (16):  y = { [(x * k_a) ↓s + n_eps]_JPEG_b } ↑s"""
import math, random, cv2, numpy as np


def random_degrade(hq, a=(0.1, 15), s=(0.8, 32), eps=(0, 20), q=(30, 100), rng=random):
    """hq: uint8 HxWx3 (BGR). Returns uint8 LQ image with the same size as hq."""
    H, W = hq.shape[:2]
    sig, sc, nz, jq = rng.uniform(*a), rng.uniform(*s), rng.uniform(*eps), int(rng.uniform(*q))
    img = hq.astype(np.float32)
    ks = 2 * math.ceil(3 * sig) + 1
    img = cv2.GaussianBlur(img, (ks, ks), sig)
    img = cv2.resize(img, (max(int(W / sc), 1), max(int(H / sc), 1)), interpolation=cv2.INTER_CUBIC)
    img = np.clip(img + np.random.randn(*img.shape).astype(np.float32) * nz, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, jq])
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return cv2.resize(img, (W, H), interpolation=cv2.INTER_CUBIC)
