from ImageShrink import ImageShrink
from NearestNeighborInterpolation import NearestNeighborInterpolation
from BilinearInterpolation import BilinearInterpolation
from BicubicInterpolation import BicubicInterpolation
from ImageRotation import ImageRotation
from CTInterpolationComparison import CTInterpolationComparison
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Run all problems
    ImageShrink.run_problem()
    NearestNeighborInterpolation.run_problem()
    BilinearInterpolation.run_problem()
    BicubicInterpolation.run_problem()
    ImageRotation.run_problem()
    CTInterpolationComparison.run_problem()
    
    # Additional demonstration with a simple test case
    print("\n=== Additional Test: Simple Interpolation Comparison ===")
    # Create a simple test image
    test_img = np.array([[1, 2], [3, 4]], dtype=float)
    print(f"Original 2x2 image:\n{test_img}")

    # Enlarge to 5x5
    enlarged_nn_test = NearestNeighborInterpolation.myNearestNeighborInterpolation(test_img, 5, 5)
    enlarged_bilinear_test = BilinearInterpolation.myBilinearInterpolation(test_img, 5, 5)
    enlarged_bicubic_test = BicubicInterpolation.myBicubicInterpolation(test_img, 5, 5)

    print(f"\nNearest Neighbor 5x5:\n{enlarged_nn_test}")
    print(f"\nBilinear 5x5:\n{enlarged_bilinear_test}")
    print(f"\nBicubic 5x5:\n{enlarged_bicubic_test}")

    # Display comparison
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    im1 = axes[0].imshow(test_img, cmap='jet', aspect='equal', interpolation='none')
    axes[0].set_title('Original 2x2')
    plt.colorbar(im1, ax=axes[0])

    im2 = axes[1].imshow(enlarged_nn_test, cmap='jet', aspect='equal', interpolation='none')
    axes[1].set_title('Nearest Neighbor 5x5')
    plt.colorbar(im2, ax=axes[1])

    im3 = axes[2].imshow(enlarged_bilinear_test, cmap='jet', aspect='equal', interpolation='none')
    axes[2].set_title('Bilinear 5x5')
    plt.colorbar(im3, ax=axes[2])

    im4 = axes[3].imshow(enlarged_bicubic_test, cmap='jet', aspect='equal', interpolation='none')
    axes[3].set_title('Bicubic 5x5')
    plt.colorbar(im4, ax=axes[3])

    plt.tight_layout()
    plt.show()

    print("\n=== Assignment Complete ===")
    print("All functions implemented:")
    print("- myImageShrink()")
    print("- myNearestNeighborInterpolation()")
    print("- myBilinearInterpolation()")
    print("- myBicubicInterpolation()")
    print("- myImageRotationUsingBilinearInterp()")
    print("- myImageRotationUsingNearestNeighborInterp()")
    print("\nNote: Make sure to place your actual image files in the data/interp/ directory")
    print("Expected files: suit.png, random.png, main.png, ct.mat")

if __name__ == "__main__":
    main()