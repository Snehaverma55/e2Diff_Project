import os
import cv2
import matplotlib.pyplot as plt

BASE = r"D:\e2diff_project\result"
blurred_path = os.path.join(BASE, "cropped_faces", "image_00.png")
restored_path = os.path.join(BASE, "restored_faces", "image_00.png")
SAVE_AS = r"D:\e2diff_project\comparison_image.png"

def load_rgb(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))

axes[0].imshow(load_rgb(blurred_path))
axes[0].set_title("Blurred image")
axes[0].axis("off")

axes[1].imshow(load_rgb(restored_path))
axes[1].set_title("Restored image")
axes[1].axis("off")

plt.tight_layout()
plt.savefig(SAVE_AS, dpi=200)
print("Saved:", SAVE_AS)
plt.show()