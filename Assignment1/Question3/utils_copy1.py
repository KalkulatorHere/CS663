import numpy as np
import matplotlib.pyplot as plt
from skimage import color, io, img_as_float
import os
from LinearContrastStretch import LinearContrastStretch
from HistogramEqualization import HistogramEqualization
from CLAHE import CLAHE
from HistogramMatching import HistogramMatching


def plot_images(images, titles, cmap=None, filename=None):
    n = len(images)
    fig, axes = plt.subplots(2, n, figsize=(5*n, 10))

    if n == 1:
        axes = axes.reshape(2, 1)

    for i in range(n):
        if images[i].ndim == 3:
            axes[0, i].imshow(images[i])
        else:
            im = axes[0, i].imshow(images[i], cmap=cmap)
            plt.colorbar(im, ax=axes[0, i])
        axes[0, i].set_title(titles[i])
        axes[0, i].axis('off')

        if images[i].ndim == 3:
            img_gray = color.rgb2gray(images[i])
            axes[1, i].hist(img_gray.flatten(), bins=256, range=(0, 1), density=True)
        else:
            axes[1, i].hist(images[i].flatten(), bins=256, range=(0, 1), density=True)
        axes[1, i].set_title(f'Histogram of {titles[i]}')
        axes[1, i].set_xlabel('Pixel Value')
        axes[1, i].set_ylabel('Frequency')

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

os.makedirs('output', exist_ok=True)
