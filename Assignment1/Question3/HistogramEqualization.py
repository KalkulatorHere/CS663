
import numpy as np
from skimage import color, exposure, img_as_float
from scipy import interpolate

class HistogramEqualization:
    def process(self, img):
        img_float = img_as_float(img)
        hsv_img = color.rgb2hsv(img_float)
        v_channel = hsv_img[:, :, 2]
        
        v_eq = exposure.equalize_hist(v_channel)
        
        hsv_eq = hsv_img.copy()
        hsv_eq[:, :, 2] = v_eq
        equalized_img = color.hsv2rgb(hsv_eq)

        return equalized_img

