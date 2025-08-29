import numpy as np

class ManualThresholder:
    @staticmethod
    def threshold(image, threshold):
        return (image > threshold).astype(np.float64)