
import numpy as np
from skimage import color, exposure, img_as_float
from scipy import interpolate

class CLAHE:
    def _clipped_hist_equalize(self, tile, clip_limit, nbins):
        hist, _ = np.histogram(tile.flatten(), bins=nbins, range=(0, 255))
        clip_threshold = max(1, int(tile.size * clip_limit / nbins))
        
        extra = 0
        for i in range(nbins):
            if hist[i] > clip_threshold:
                extra += hist[i] - clip_threshold
                hist[i] = clip_threshold

        redist_val = extra // nbins
        hist += redist_val

        cdf = hist.cumsum()
        if cdf[-1] == 0:
            mapping = np.arange(nbins)
        else:
            mapping = (cdf * 255.0) / cdf[-1]

        equalized_tile = mapping[tile].astype(np.uint8)
        return equalized_tile

    def process(self, img, nbins=256, tile_size=64, clip_limit=0.03):
        img_float = img_as_float(img)
        hsv_img = color.rgb2hsv(img_float)
        v_channel = hsv_img[:, :, 2]
        
        v_8bit = (v_channel * 255).astype(np.uint8)
        v_clahe = v_8bit.copy()
        height, width = v_8bit.shape

        for i in range(0, height, tile_size):
            for j in range(0, width, tile_size):
                i_end = min(i + tile_size, height)
                j_end = min(j + tile_size, width)
                tile = v_8bit[i:i_end, j:j_end]
                processed_tile = self._clipped_hist_equalize(tile, clip_limit, nbins)
                v_clahe[i:i_end, j:j_end] = processed_tile

        v_clahe_normalized = v_clahe / 255.0
        hsv_clahe = hsv_img.copy()
        hsv_clahe[:, :, 2] = v_clahe_normalized
        clahe_img = color.hsv2rgb(hsv_clahe)

        return clahe_img

