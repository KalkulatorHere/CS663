import numpy as np
import matplotlib.pyplot as plt
from ImageUtils import ImageUtils

class BicubicInterpolation:
    @staticmethod
    def cubic_kernel(x):
        """Cubic interpolation kernel"""
        x = abs(x)
        if x <= 1:
            return 1 - 2*x*x + x*x*x
        elif x <= 2:
            return 4 - 8*x + 5*x*x - x*x*x
        else:
            return 0

    @staticmethod
    def myBicubicInterpolation(image, new_rows, new_cols):
        """Resize image using bicubic interpolation"""
        M, N = image.shape
        resized = np.zeros((new_rows, new_cols))
        
        for i in range(new_rows):
            for j in range(new_cols):
                # Map new coordinates to original coordinates
                orig_row = i * (M - 1) / (new_rows - 1)
                orig_col = j * (N - 1) / (new_cols - 1)
                
                # Find center of 4x4 neighborhood
                center_row = int(np.floor(orig_row))
                center_col = int(np.floor(orig_col))
                
                val = 0
                for m in range(-1, 3):
                    for n in range(-1, 3):
                        row_idx = center_row + m
                        col_idx = center_col + n
                        
                        # Handle boundaries
                        row_idx = max(0, min(row_idx, M - 1))
                        col_idx = max(0, min(col_idx, N - 1))
                        
                        # Calculate cubic weights
                        weight_row = BicubicInterpolation.cubic_kernel(orig_row - (center_row + m))
                        weight_col = BicubicInterpolation.cubic_kernel(orig_col - (center_col + n))
                        
                        val += weight_row * weight_col * image[row_idx, col_idx]
                
                resized[i, j] = val
        
        return resized

    @staticmethod
    def run_problem():
        print("\n=== Problem 1(d): Bicubic Interpolation ===")
        try:
            random_img = ImageUtils.load_image("data/interp/random.png", (10, 10), False)
            
            M, N = random_img.shape
            new_rows = 300 * (M - 1) + 1
            new_cols = 300 * (N - 1) + 1
            
            enlarged_bicubic = BicubicInterpolation.myBicubicInterpolation(random_img, new_rows, new_cols)
            
            # Display results
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            im1 = axes[0].imshow(random_img, cmap='jet', aspect='equal')
            axes[0].set_title(f'Original Image ({M}x{N})')
            plt.colorbar(im1, ax=axes[0])
            
            im2 = axes[1].imshow(enlarged_bicubic, cmap='jet', aspect='equal')
            axes[1].set_title(f'Bicubic Interpolation ({new_rows}x{new_cols})')
            plt.colorbar(im2, ax=axes[1])
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error in Problem 1(d): {e}")