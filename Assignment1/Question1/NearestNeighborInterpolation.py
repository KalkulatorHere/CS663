import numpy as np
import matplotlib.pyplot as plt
from ImageUtils import ImageUtils

class NearestNeighborInterpolation:
    @staticmethod
    def myNearestNeighborInterpolation(image, new_rows, new_cols):
        """Resize image using nearest neighbor interpolation"""
        M, N = image.shape
        resized = np.zeros((new_rows, new_cols))
        
        for i in range(new_rows):
            for j in range(new_cols):
                # Map new coordinates to original coordinates
                orig_row = i * (M - 1) / (new_rows - 1)
                orig_col = j * (N - 1) / (new_cols - 1)
                
                # Find nearest neighbor
                row_idx = int(round(orig_row))
                col_idx = int(round(orig_col))
                
                # Clamp to valid indices
                row_idx = min(max(row_idx, 0), M - 1)
                col_idx = min(max(col_idx, 0), N - 1)
                
                resized[i, j] = image[row_idx, col_idx]
        
        return resized

    @staticmethod
    def run_problem():
        print("\n=== Problem 1(b): Nearest Neighbor Interpolation ===")
        try:
            random_img = ImageUtils.load_image("data/interp/random.png", (10, 10), False)
            
            M, N = random_img.shape
            new_rows = 300 * (M - 1) + 1
            new_cols = 300 * (N - 1) + 1
            
            enlarged_nn = NearestNeighborInterpolation.myNearestNeighborInterpolation(random_img, new_rows, new_cols)
            
            # Display results
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            im1 = axes[0].imshow(random_img, cmap='jet', aspect='equal')
            axes[0].set_title(f'Original Image ({M}x{N})')
            plt.colorbar(im1, ax=axes[0])
            
            im2 = axes[1].imshow(enlarged_nn, cmap='jet', aspect='equal')
            axes[1].set_title(f'Nearest Neighbor ({new_rows}x{new_cols})')
            plt.colorbar(im2, ax=axes[1])
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error in Problem 1(b): {e}")