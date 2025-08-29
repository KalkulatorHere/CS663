import numpy as np
import matplotlib.pyplot as plt
from ImageUtils import ImageUtils

class BilinearInterpolation:
    @staticmethod
    def myBilinearInterpolation(image, new_rows, new_cols):
        """Resize image using bilinear interpolation"""
        M, N = image.shape
        resized = np.zeros((new_rows, new_cols))
        
        for i in range(new_rows):
            for j in range(new_cols):
                orig_row = i * (M - 1) / (new_rows - 1)
                orig_col = j * (N - 1) / (new_cols - 1)
                
                row1 = int(np.floor(orig_row))
                row2 = min(row1 + 1, M - 1)
                col1 = int(np.floor(orig_col))
                col2 = min(col1 + 1, N - 1)
                
                dr = orig_row - row1
                dc = orig_col - col1
                
                val = (1 - dr) * (1 - dc) * image[row1, col1] + \
                      dr * (1 - dc) * image[row2, col1] + \
                      (1 - dr) * dc * image[row1, col2] + \
                      dr * dc * image[row2, col2]
                
                resized[i, j] = val
        
        return resized

    @staticmethod
    def run_problem():
        print("\n=== Problem 1(c): Bilinear Interpolation ===")
        try:
            random_img = ImageUtils.load_image("data/interp/random.png", (10, 10), False)
            
            M, N = random_img.shape
            new_rows = 300 * (M - 1) + 1
            new_cols = 300 * (N - 1) + 1
            
            enlarged_bilinear = BilinearInterpolation.myBilinearInterpolation(random_img, new_rows, new_cols)
            
            # Display results
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            im1 = axes[0].imshow(random_img, cmap='jet', aspect='equal')
            axes[0].set_title(f'Original Image ({M}x{N})')
            plt.colorbar(im1, ax=axes[0])
            
            im2 = axes[1].imshow(enlarged_bilinear, cmap='jet', aspect='equal')
            axes[1].set_title(f'Bilinear Interpolation ({new_rows}x{new_cols})')
            plt.colorbar(im2, ax=axes[1])
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error in Problem 1(c): {e}")