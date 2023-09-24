## Get a directory of the image library.
## Make a image search engine for finding the appropriate image for specific pixel
## Analyze the source image.
## Make a mixture of image.

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from pathlib import Path
import glob 
import random
import json
import tqdm

import sys, os
# sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'COCO'))

from coco import COCO

class ImageGridDrawer():
    def __init__(self, w, h):
        self.w = w
        self.h = h

        fig_size = (100, int(100*h/w))
        fig, axes = plt.subplots(nrows=h, ncols=w, fig_size=fig_size)
        self.result = axes

    def _init(self):
        assert len(self.images) >= self.w*self.h, "Please Check for the target width and height"
        self.sample_image(self.w*self.h)

    def sample_image(self, target_size):
        self.images = random.sample(self.images, target_size)
        self.total = len(self.images)

    def image_w_h(self):
        target_img = random.choice(self.images)
        temp = Image.open(target_img)
        w, h = temp.size
        return w, h

    def _draw(self, out):
        self._init()
        one_image_w, one_image_h = self.image_w_h()
        max_rows, max_cols = self.h, self.w

        # fig, axes = plt.subplots(nrows=max_rows, ncols=max_cols, figsize=(20,20))
        for idx, image in enumerate(self.images):
            row = idx // max_cols
            col = idx % max_cols
            im = plt.imread(image)
            self.draw_pixel(row, col, im)

        plt.subplots_adjust(wspace=0, hspace=0)
        plt.savefig(out)

    def draw_pixel(self, row, col, im):
        self.axes[row, col].axis("off")
        self.axes[row, col].imshow(im, cmap="gray", aspect="fit")
    
    @classmethod
    def draw_from_lib(cls, lib_path, w, h):
        cls.lib_path = Path(lib_path)
        cls.images = glob.glob(str(lib_path) / "*")
        return cls(w, h)
        
    def draw(self, image_paths, out="result.png"):
        if image_paths != None:
            self.images = image_paths
        self._draw(out)


def seen_to_grid(seen_paths):
    seens = []
    
    coco_indexer = COCO("../coco_viewer_api_server/captions_val2014.json")

    for _seen in seen_paths:
        with open(_seen, "r", encoding="utf-8") as f:
            data = json.load(f)
            seens.extend(data["seen"])

    seen_images = []
    prefix = "/Volumes/T7/coco_viewer_web/static/val2014/"
    for seen in seens:
        try: 
            data, ann = coco_indexer.get_image_by_id(seen)
            src = prefix + data["file_name"]
            seen_images.append(src)
        except: pass

    print(len(seen_images))

    drawer = ImageGridDrawer(w=40, h=20)
    drawer.draw(image_paths=seen_images, w=40, h=20, out="result.png")


class ImageAnalyzer():
    def __init__(self, lib_files, block_size):
        self.lib_files = lib_files
        self.block_size = block_size

        # prepare image to get a representative pixel. 
        self._prepare()

    @classmethod
    def from_lib(cls, lib_path):
       lib_files = str(Path(lib_path) / "*")
       return cls(lib_files=lib_files)

    @staticmethod
    def resizeImage(im_path, width, height):
        return ImageOps.fit(
            Image.open(im_path).convert("RGB"), (width, height), Image.Resampling.LANCZOS
        )

    @staticmethod
    def loadImage(input_dir):
        return Image.open(input_dir)

    @staticmethod
    def getAvgPixel(image):
        hsv_colors = np.array(image.convert("HSV"), dtype=np.float32) / 255.
        hsv_colors = hsv_colors.reshape(-1, hsv_colors.shape[-1])
        hsv_avg = np.mean(hsv_colors, axis=0)
        return hsv_avg

    def _prepare(self):
        temp = []
        data = np.empty((len(self.lib_files), 3), dtype=np.float32)
        for i, image_path in enumerate(tqdm.tqdm(self.lib_files, desc="calc...")):
            try:
                img = ImageAnalyzer.loadImage(image_path)
                hsv_avg = ImageAnalyzer.getAvgPixel(img)
                data[i, :] = hsv_avg
                temp.append(image_path)
            except:
                print(f"Error During Loading the Image path from {image_path}")
                pass
        
        self.data = data
        self.image_path = temp

    @staticmethod
    def cosine_similarity(x, y):
        return np.dot(x, y) / (np.linalg.norm(x, axis=1) * np.linalg.norm(y) + 1e-6)

    @staticmethod
    def normalize(dist):
        return np.exp(dist-np.max(dist)) / sum(np.exp(dist-np.max(dist)))

    def findClosest(self, target):
        sim = ImageAnalyzer.cosine_similarity(self.data, target)
        prob = ImageAnalyzer.normalize(sim)

        random_selected = np.random.multinomial(1, prob)
        selected_index = np.argmax(random_selected)
        selected_image = self.image_path(selected_index)

        return ImageAnalyzer.resizeImage(ImageAnalyzer.loadImage(selected_image), self.BLOCK_SIZE)


def drawPuzzle(imageAnalyzer:ImageAnalyzer, target_image, BLOCK_SIZE=5): 
    width, height = target_image.size
    result = Image.new("RGB", target_image.size, (255, 255, 255))
    width, height = width // BLOCK_SIZE, height // BLOCK_SIZE

    for j in range(height):
        for i in range(width):
            try: 
                x, y = i*BLOCK_SIZE, j*BLOCK_SIZE   
                target_block = target_image.crop((x, y, x+BLOCK_SIZE, y+BLOCK_SIZE))
                target_color = ImageAnalyzer.getAvgPixel(target_block)
                im = imageAnalyzer.findCloset(target_color)
                result.paste(im, (x, y))
            except:
                pass
    return result

if __name__ == "__main__":
    seen_data_paths = ["../coco_viewer_api_server/seen.json"]

    seen_to_grid(seen_data_paths)
    

    
    
        