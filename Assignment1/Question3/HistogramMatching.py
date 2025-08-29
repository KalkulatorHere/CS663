import numpy as np
from skimage import color, exposure, img_as_float
from scipy import interpolate
class HistogramMatching:
    def process(self, img, ref_img, nbins=256):
        def _compute_mapping(src_vals, ref_vals):
            src_hist, _ = np.histogram(src_vals, bins=nbins, range=(0, 255))
            ref_hist, _ = np.histogram(ref_vals, bins=nbins, range=(0, 255))

            src_cdf = src_hist.cumsum().astype(np.float64)
            ref_cdf = ref_hist.cumsum().astype(np.float64)

            if src_cdf[-1] == 0: src_cdf[-1] = 1
            if ref_cdf[-1] == 0: ref_cdf[-1] = 1

            src_cdf /= src_cdf[-1]
            ref_cdf /= ref_cdf[-1]

            mapping = np.zeros(nbins, dtype=np.int32)
            j = 0
            for i in range(nbins):
                while j < nbins and ref_cdf[j] < src_cdf[i]:
                    j += 1
                mapping[i] = j if j < nbins else nbins - 1
            return mapping

        img_mask = np.any(img > 5, axis=2)
        ref_mask = np.any(ref_img > 5, axis=2)

        yuv_matrix = np.array([[0.299, 0.587, 0.114],
                               [-0.147, -0.289, 0.436],
                               [0.615, -0.515, -0.100]])
        yuv_inv = np.linalg.inv(yuv_matrix)

        yuv_img = np.dot(img, yuv_matrix.T)
        yuv_ref = np.dot(ref_img, yuv_matrix.T)

        yuv_img = np.clip(yuv_img, 0, 255)
        yuv_ref = np.clip(yuv_ref, 0, 255)

        yuv_matched = np.zeros_like(yuv_img, dtype=np.float64)

        for channel in range(3):
            src_channel = yuv_img[:, :, channel].astype(np.uint32)
            ref_channel = yuv_ref[:, :, channel].astype(np.uint32)

            mapping = _compute_mapping(src_channel[img_mask], ref_channel[ref_mask])
            scaled_src = (src_channel * (nbins - 1) / 255).astype(np.int32)
            mapped_bins = mapping[scaled_src]
            matched_channel = (mapped_bins * 255 / (nbins - 1))
            yuv_matched[:, :, channel] = matched_channel

        rgb_out = np.dot(yuv_matched, yuv_inv.T)
        matched_img = np.clip(rgb_out, 0, 255).astype(np.uint8)

        return matched_img