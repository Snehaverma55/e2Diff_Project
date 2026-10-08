import copy, cv2, numpy as np, torch


class EMA:
    """Exponential moving average of model weights (decay 0.999 as in the paper)."""
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval().requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for e, p in zip(self.shadow.parameters(), model.parameters()):
            e.mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)
        for e, b in zip(self.shadow.buffers(), model.buffers()):
            e.copy_(b)


def bgr_to_tensor(img):
    """uint8 HxWx3 BGR -> float CHW RGB in [-1,1]"""
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(rgb).permute(2, 0, 1) * 2 - 1


def tensor_to_bgr(t):
    """float CHW (or 1CHW) in [-1,1] -> uint8 HxWx3 BGR"""
    if t.dim() == 4:
        t = t[0]
    rgb = ((t.clamp(-1, 1) + 1) / 2 * 255).round().byte().permute(1, 2, 0).cpu().numpy()
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def count_params(m):
    return sum(p.numel() for p in m.parameters()) / 1e6


IMG_EXT = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
