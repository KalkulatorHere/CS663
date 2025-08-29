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
    

if __name__ == "__main__":
    main()