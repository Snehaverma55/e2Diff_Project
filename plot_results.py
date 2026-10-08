import os
import cv2
import matplotlib.pyplot as plt

BASE = r"D:\e2diff_project"
SAVE_AS = os.path.join(BASE, "comparison.png")

# (row label, folder containing cropped_faces and restored_faces, file name)
items = [
    ("Screenshot 085649", os.path.join(BASE, "results"), "Screenshot 2026-08-24 085649_00.png"),
    ("Screenshot 085705", os.path.join(BASE, "results"), "Screenshot 2026-08-24 085705_00.png"),
    ("image",             os.path.join(BASE, "result"),  "image_00.png"),
]

def load_rgb(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig, axes = plt.subplots(len(items), 2, figsize=(8, 4 * len(items)), squeeze=False)

for r, (label, folder, fname) in enumerate(items):
    blurred = os.path.join(folder, "cropped_faces", fname)
    restored = os.path.join(folder, "restored_faces", fname)
    axes[r][0].imshow(load_rgb(blurred))
    axes[r][0].set_title("Blurred image")
    axes[r][1].imshow(load_rgb(restored))
    axes[r][1].set_title("Restored image")
    axes[r][0].text(-0.05, 0.5, label, transform=axes[r][0].transAxes,
                    rotation=90, va="center", ha="right", fontsize=10)
    axes[r][0].axis("off")
    axes[r][1].axis("off")

plt.tight_layout()
plt.savefig(SAVE_AS, dpi=200)
print("Saved:", SAVE_AS)
plt.show()