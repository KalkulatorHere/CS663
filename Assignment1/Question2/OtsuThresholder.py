import numpy as np

class OtsuThresholder:
    @staticmethod
    def threshold(image):
        hist, bin_edges = np.histogram(image * 255, bins=256, range=(0, 255))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        p = hist / hist.sum()
        best_threshold = 0
        best_variance = 0

        for t in range(1, 255):
            w0 = p[:t].sum()
            w1 = p[t:].sum()
            if w0 == 0 or w1 == 0:
                continue
            mu0 = np.sum(p[:t] * bin_centers[:t]) / w0 if w0 > 0 else 0
            mu1 = np.sum(p[t:] * bin_centers[t:]) / w1 if w1 > 0 else 0
            variance = w0 * w1 * (mu0 - mu1) ** 2
            if variance > best_variance:
                best_variance = variance
                best_threshold = bin_centers[t]

        best_threshold_normalized = best_threshold / 255.0
        return (image > best_threshold_normalized).astype(np.float64), best_threshold_normalized