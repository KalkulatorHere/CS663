import numpy as np
from scipy import ndimage

class AdaptiveThresholder:
    @staticmethod
    def threshold(image, block_size=15, c=0.05):
        if block_size % 2 == 0:
            block_size += 1
        local_mean = ndimage.uniform_filter(image, size=block_size)
        thresholded = (image > (local_mean - c)).astype(np.float64)
        return thresholded, local_mean