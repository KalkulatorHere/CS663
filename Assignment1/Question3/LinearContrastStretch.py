import numpy as np
from skimage import color, exposure, img_as_float
from scipy import interpolate

class LinearContrastStretch:
    def process(self, img):
        img_float = img_as_float(img)
        hsv_img = color.rgb2hsv(img_float)
        v_channel = hsv_img[:, :, 2]
        
        v_min = np.min(v_channel)
        v_max = np.max(v_channel)

        if v_max > v_min:
            v_stretched = (v_channel - v_min) / (v_max - v_min)
        else:
            v_stretched = v_channel

        hsv_stretched = hsv_img.copy()
        hsv_stretched[:, :, 2] = v_stretched
        stretched_img = color.hsv2rgb(hsv_stretched)

        return stretched_img

