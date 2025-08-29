import numpy as np
from PIL import Image
import os

class ImageLoader:
    @staticmethod
    def load_images(image_files):
        images = {}
        for name, path in image_files.items():
            if os.path.exists(path):
                img = Image.open(path)
                if img.mode != 'L':
                    img = img.convert('L')
                img_array = np.array(img).astype(np.float64)
                if img_array.max() > 1:
                    img_array = img_array / 255.0
                images[name] = img_array
            else:
                print(f"Warning: {path} not found")
                images[name] = np.random.rand(100, 100)
        return images