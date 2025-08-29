import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import os
from ImageUtils import ImageUtils
from NearestNeighborInterpolation import NearestNeighborInterpolation
from BilinearInterpolation import BilinearInterpolation
from BicubicInterpolation import BicubicInterpolation

class CTInterpolationComparison:
    @staticmethod
    def run_problem():
        print("\n=== Problem 1(f): CT Image Interpolation Comparison ===")
        try:
            ct_path = "data/interp/ct.mat"
            if os.path.exists(ct_path):
                ct_data = scipy.io.loadmat(ct_path)
                original_ct = ct_data['original'].astype(float)
                subsampled_ct = ct_data['subsampled'].astype(float)
            else:
                print("Creating sample CT images (ct.mat not found)")
                original_ct = np.random.rand(100, 100) * 255
                subsampled_ct = original_ct[::4, ::4]
            
            target_rows, target_cols = original_ct.shape
            sub_rows, sub_cols = subsampled_ct.shape
            
            enlarged_nn = NearestNeighborInterpolation.myNearestNeighborInterpolation(subsampled_ct, target_rows, target_cols)
            enlarged_bilinear = BilinearInterpolation.myBilinearInterpolation(subsampled_ct, target_rows, target_cols)
            enlarged_bicubic = BicubicInterpolation.myBicubicInterpolation(subsampled_ct, target_rows, target_cols)
            
            rmse_nn = ImageUtils.calculate_rmse(original_ct, enlarged_nn)
            rmse_bilinear = ImageUtils.calculate_rmse(original_ct, enlarged_bilinear)
            rmse_bicubic = ImageUtils.calculate_rmse(original_ct, enlarged_bicubic)
            
            print(f"RMSE Results:")
            print(f"Nearest Neighbor: {rmse_nn:.4f}")
            print(f"Bilinear: {rmse_bilinear:.4f}")
            print(f"Bicubic: {rmse_bicubic:.4f}")
            
            fig, axes = plt.subplots(2, 4, figsize=(20, 10))
            
            vmin = min(original_ct.min(), enlarged_nn.min(), enlarged_bilinear.min(), enlarged_bicubic.min())
            vmax = max(original_ct.max(), enlarged_nn.max(), enlarged_bilinear.max(), enlarged_bicubic.max())
            
            im1 = axes[0,0].imshow(original_ct, cmap='jet', aspect='equal', vmin=vmin, vmax=vmax)
            axes[0,0].set_title('Original CT')
            plt.colorbar(im1, ax=axes[0,0])
            
            im2 = axes[0,1].imshow(enlarged_nn, cmap='jet', aspect='equal', vmin=vmin, vmax=vmax)
            axes[0,1].set_title(f'Nearest Neighbor (RMSE: {rmse_nn:.2f})')
            plt.colorbar(im2, ax=axes[0,1])
            
            im3 = axes[0,2].imshow(enlarged_bilinear, cmap='jet', aspect='equal', vmin=vmin, vmax=vmax)
            axes[0,2].set_title(f'Bilinear (RMSE: {rmse_bilinear:.2f})')
            plt.colorbar(im3, ax=axes[0,2])
            
            im4 = axes[0,3].imshow(enlarged_bicubic, cmap='jet', aspect='equal', vmin=vmin, vmax=vmax)
            axes[0,3].set_title(f'Bicubic (RMSE: {rmse_bicubic:.2f})')
            plt.colorbar(im4, ax=axes[0,3])
            
            diff_nn = original_ct - enlarged_nn
            diff_bilinear = original_ct - enlarged_bilinear
            diff_bicubic = original_ct - enlarged_bicubic
            
            diff_vmin = min(diff_nn.min(), diff_bilinear.min(), diff_bicubic.min())
            diff_vmax = max(diff_nn.max(), diff_bilinear.max(), diff_bicubic.max())
            
            axes[1,0].axis('off')
            
            im5 = axes[1,1].imshow(diff_nn, cmap='jet', aspect='equal', vmin=diff_vmin, vmax=diff_vmax)
            axes[1,1].set_title('Difference: Original - NN')
            plt.colorbar(im5, ax=axes[1,1])
            
            im6 = axes[1,2].imshow(diff_bilinear, cmap='jet', aspect='equal', vmin=diff_vmin, vmax=diff_vmax)
            axes[1,2].set_title('Difference: Original - Bilinear')
            plt.colorbar(im6, ax=axes[1,2])
            
            im7 = axes[1,3].imshow(diff_bicubic, cmap='jet', aspect='equal', vmin=diff_vmin, vmax=diff_vmax)
            axes[1,3].set_title('Difference: Original - Bicubic')
            plt.colorbar(im7, ax=axes[1,3])
            
            plt.tight_layout()
            plt.show()
            
            print("\nQuality Analysis:")
            print("From the enlarged images with same colormap limits:")
            print("- Bicubic interpolation typically produces the smoothest results")
            print("- Bilinear interpolation provides good balance between smoothness and computational cost")
            print("- Nearest neighbor produces blocky artifacts but preserves sharp edges")
            print("\nFrom the difference images with same colormap limits:")
            print("- Bicubic typically has the lowest reconstruction error")
            print("- The difference images show where each method introduces artifacts")
            
        except Exception as e:
            print(f"Error in Problem 1(f): {e}")