import os, random, cv2, torch
from torch.utils.data import Dataset
from .degradation import random_degrade
from .utils import bgr_to_tensor, IMG_EXT


def list_images(folder):
    return sorted(os.path.join(r, f) for r, _, fs in os.walk(folder) for f in fs if f.lower().endswith(IMG_EXT))


class FaceFolder(Dataset):
    """HQ faces (FFHQ). mode='hq' -> HQ tensor ; mode='pair' -> (LQ, HQ) with random degradation."""
    def __init__(self, root, size=512, mode='hq', crop=None):
        self.files, self.size, self.mode, self.crop = list_images(root), size, mode, crop
        assert self.files, f'no images found in {root}'

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = cv2.imread(self.files[i], cv2.IMREAD_COLOR)
        if img.shape[0] != self.size:
            img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA)
        if random.random() < 0.5:
            img = img[:, ::-1].copy()
        if self.mode == 'hq':
            return bgr_to_tensor(img)
        lq, hq = bgr_to_tensor(random_degrade(img)), bgr_to_tensor(img)
        if self.crop and self.crop < self.size:
            y, x = random.randint(0, self.size - self.crop), random.randint(0, self.size - self.crop)
            lq, hq = lq[:, y:y + self.crop, x:x + self.crop], hq[:, y:y + self.crop, x:x + self.crop]
        return lq, hq
