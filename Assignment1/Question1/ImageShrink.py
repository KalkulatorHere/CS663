import numpy as np
import matplotlib.pyplot as plt
from ImageUtils import ImageUtils

class ImageShrink:
    @staticmethod
    def myImageShrink(image, d):
        """Shrink image by factor d using subsampling"""
        return image[::d, ::d]

    @staticmethod
    def run_problem():
        print("=== Problem 1(a): Image Shrinking ===")
        try:
            suit_img = ImageUtils.load_image("data/interp/suit.png")
            
            shrunk_d2 = ImageShrink.myImageShrink(suit_img, 2)
            shrunk_d3 = ImageShrink.myImageShrink(suit_img, 3)
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            im1 = axes[0].imshow(suit_img, cmap='gray', aspect='equal')
            axes[0].set_title('Original Image')
            plt.colorbar(im1, ax=axes[0])
            
            im2 = axes[1].imshow(shrunk_d2, cmap='gray', aspect='equal')
            axes[1].set_title('Subsampled (d=2)')
            plt.colorbar(im2, ax=axes[1])
            
            im3 = axes[2].imshow(shrunk_d3, cmap='gray', aspect='equal')
            axes[2].set_title('Subsampled (d=3)')
            plt.colorbar(im3, ax=axes[2])
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error in Problem 1(a): {e}")