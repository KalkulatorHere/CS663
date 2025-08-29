import matplotlib.pyplot as plt
from pathlib import Path

class ImageDisplayer:
    def __init__(self, output_dir="output_thresholding"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def display_images(self, images, titles, cmap='gray', figsize=(15, 10), filename=None):
        n = len(images)
        fig, axes = plt.subplots(1, n, figsize=figsize)
        if n == 1:
            axes = [axes]

        for i, (img, title) in enumerate(zip(images, titles)):
            im = axes[i].imshow(img, cmap=cmap)
            axes[i].set_title(title)
            plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

        plt.tight_layout()
        if filename:
            plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.show()