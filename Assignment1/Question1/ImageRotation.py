import numpy as np
import matplotlib.pyplot as plt
from ImageUtils import ImageUtils

class ImageRotation:
    @staticmethod
    def myImageRotationUsingBilinearInterp(image, angle_degrees):
        """Rotate image using bilinear interpolation"""
        angle_rad = np.deg2rad(angle_degrees)
        M, N = image.shape
        center_row, center_col = M // 2, N // 2
        
        rotated = np.zeros_like(image)
        
        for i in range(M):
            for j in range(N):
                # Translate to center
                y = i - center_row
                x = j - center_col
                
                # Rotate coordinates (inverse rotation)
                orig_x = x * np.cos(angle_rad) + y * np.sin(angle_rad)
                orig_y = -x * np.sin(angle_rad) + y * np.cos(angle_rad)
                
                # Translate back
                orig_row = orig_y + center_row
                orig_col = orig_x + center_col
                
                # Check bounds
                if 0 <= orig_row < M-1 and 0 <= orig_col < N-1:
                    # Bilinear interpolation
                    row1 = int(np.floor(orig_row))
                    row2 = row1 + 1
                    col1 = int(np.floor(orig_col))
                    col2 = col1 + 1
                    
                    dr = orig_row - row1
                    dc = orig_col - col1
                    
                    val = (1 - dr) * (1 - dc) * image[row1, col1] + \
                          dr * (1 - dc) * image[row2, col1] + \
                          (1 - dr) * dc * image[row1, col2] + \
                          dr * dc * image[row2, col2]
                    
                    rotated[i, j] = val
        
        return rotated

    @staticmethod
    def myImageRotationUsingNearestNeighborInterp(image, angle_degrees):
        """Rotate image using nearest neighbor interpolation"""
        angle_rad = np.deg2rad(angle_degrees)
        M, N = image.shape
        center_row, center_col = M // 2, N // 2
        
        rotated = np.zeros_like(image)
        
        for i in range(M):
            for j in range(N):
                # Translate to center
                y = i - center_row
                x = j - center_col
                
                # Rotate coordinates (inverse rotation)
                orig_x = x * np.cos(angle_rad) + y * np.sin(angle_rad)
                orig_y = -x * np.sin(angle_rad) + y * np.cos(angle_rad)
                
                # Translate back
                orig_row = int(round(orig_y + center_row))
                orig_col = int(round(orig_x + center_col))
                
                # Check bounds and assign
                if 0 <= orig_row < M and 0 <= orig_col < N:
                    rotated[i, j] = image[orig_row, orig_col]
        
        return rotated

    @staticmethod
    def run_problem():
        print("\n=== Problem 1(e): Image Rotation ===")
        try:
            main_img = ImageUtils.load_image("data/interp/main.png", (200, 200), False)
            
            # Create a sample image with a slanted line
            if main_img.shape == (200, 200):
                for i in range(200):
                    for j in range(200):
                        if abs((j - 100) - 0.5 * (i - 100)) < 2:
                            main_img[i, j] = 255
            
            # Estimate rotation angle needed
            rotation_angle = 15  # degrees to make lamp post vertical
            
            rotated_bilinear = ImageRotation.myImageRotationUsingBilinearInterp(main_img, rotation_angle)
            rotated_nn = ImageRotation.myImageRotationUsingNearestNeighborInterp(main_img, rotation_angle)
            
            # Display results
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            im1 = axes[0].imshow(main_img, cmap='gray', aspect='equal')
            axes[0].set_title('Original Image')
            plt.colorbar(im1, ax=axes[0])
            
            im2 = axes[1].imshow(rotated_bilinear, cmap='gray', aspect='equal')
            axes[1].set_title(f'Rotated (Bilinear, {rotation_angle}°)')
            plt.colorbar(im2, ax=axes[1])
            
            im3 = axes[2].imshow(rotated_nn, cmap='gray', aspect='equal')
            axes[2].set_title(f'Rotated (Nearest Neighbor, {rotation_angle}°)')
            plt.colorbar(im3, ax=axes[2])
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error in Problem 1(e): {e}")