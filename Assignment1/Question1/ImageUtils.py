import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import scipy.io
import os

class ImageUtils:
    @staticmethod
    def calculate_rmse(img1, img2):
        """Calculate Root Mean Square Error"""
        return np.sqrt(np.mean((img1 - img2) ** 2))

    @staticmethod
    def display_image_with_colorbar(image, title, colormap='viridis', figsize=(8, 6)):
        """Display image with colorbar"""
        plt.figure(figsize=figsize)
        plt.imshow(image, cmap=colormap, aspect='equal')
        plt.colorbar()
        plt.title(title)
        plt.xlabel('Pixel Column')
        plt.ylabel('Pixel Row')
        plt.show()

    @staticmethod
    def load_image(path, default_size=(256, 256), pattern=True):
        """Load image or create sample if not found"""
        if os.path.exists(path):
            return np.array(Image.open(path).convert('L')).astype(float)
        else:
            print(f"Creating sample image ({path} not found)")
            if pattern:
                img = np.random.rand(*default_size) * 255
                x, y = np.meshgrid(np.arange(default_size[0]), np.arange(default_size[1]))
                pattern = np.sin(0.3 * x) * np.sin(0.3 * y) + np.sin(0.5 * x) * np.sin(0.7 * y)
                return 128 + 50 * pattern
            else:
                return np.random.rand(*default_size) * 255