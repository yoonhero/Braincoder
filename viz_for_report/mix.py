## Get a directory of the image library.
## Make a image search engine for finding the appropriate image for specific pixel
## Analyze the source image.
## Make a mixture of image.

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import glob 
import random
import json

import sys, os
# sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'COCO'))

from coco import COCO

class ImageGridDrawer():
    def _init(self, w, h):
        assert len(self.images) >= w*h, "Please Check for the target width and height"
        self.sample_image(w*h)

    def sample_image(self, target_size):
        self.images = random.sample(self.images, target_size)
        self.total = len(self.images)

    def image_w_h(self):
        target_img = random.choice(self.images)
        temp = Image.open(target_img)
        w, h = temp.size
        return w, h

    def _draw(self, w, h, out):
        self._init(w, h)
        one_image_w, one_image_h = self.image_w_h()
        max_rows, max_cols = h, w

        fig, axes = plt.subplots(nrows=max_rows, ncols=max_cols, figsize=(20,20))
        for idx, image in enumerate(self.images):
            row = idx // max_cols
            col = idx % max_cols
            axes[row, col].axis("off")
            im = plt.imread(image)
            axes[row, col].imshow(im, cmap="gray", aspect="auto")
        plt.subplots_adjust(wspace=.05, hspace=.05)

        plt.savefig(out)
    
    def draw_from_lib(self, lib_path, w, h, out="result.png"):
        self.lib_path = Path(lib_path)
        self.images = glob.glob(str(lib_path) / "*")
        self._draw(w, h, out)
        
    def draw(self, image_paths, w, h, out="result.png"):
        self.images = image_paths
        self._draw(w, h, out)


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

    drawer = ImageGridDrawer()
    drawer.draw(image_paths=seen_images, w=40, h=20, out="result.png")


if __name__ == "__main__":
    seen_data_paths = ["../coco_viewer_api_server/seen.json"]

    seen_to_grid(seen_data_paths)
    

    
    
        