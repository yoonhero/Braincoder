import os
# from pycocotools.coco import COCO
# from pycocoevalcap.eval import COCOEvalCap
import matplotlib.pyplot as plt
import skimage.io as io
from functools import lru_cache
from collections import defaultdict
import time
import random

import json
from json import encoder

class COCO():
    def __init__(self, annFile):
        start = time.time()
        with open(annFile, "r") as f:
            dataset = json.load(f)
        self.dataset = dataset
        print(f"Load the Dataset in {time.time() - start}s!!")       

        self._prepare()


    def _prepare(self):
        anns, imgs, cats = {}, {}, {}
        imgToAnns = defaultdict(list)

        if "annotations" in self.dataset:
            for ann in self.dataset["annotations"]:
                anns[ann["id"]] = ann
                imgToAnns[ann["image_id"]].append(ann)
        if "images" in self.dataset:
            for image in self.dataset["images"]:
                imgs[image["id"]] = image
        if 'categories' in self.dataset:
            for cat in self.dataset['categories']:
                cats[cat['id']] = cat
        else:
            for key in imgToAnns.keys():
                ann = imgToAnns[key]
                caption = ann[0]["caption"]
                cats[key] = caption

        self.anns = anns
        self.imgs = imgs
        self.cats = cats
        self.imgToAnns = imgToAnns
        self.ids = self.get_image_ids()

    @lru_cache
    def get_image_ids(self):
        return self.imgs.keys()

    def get_image_by_id(self, image_id):
        image, ann = None, None
        try:
            image = self.imgs[image_id]
            ann = self.imgToAnns[image_id]
        except KeyError:
            print("Please check the image id.")

        return image, ann
    
    def sampling(self, ids):
        image_id = random.sample(ids, 1)[0]

        return self.get_image_by_id(image_id)

    def sample_with_topic(self, keywords):
        results = []
        for item in self.cats.items():
            caption = item[1].lower().split(" ")
            for keyword in keywords:
                if keyword.lower() in caption:
                    results.append(item[0])
        return results
    
    def find_image(self, filename):
        for item in self.imgs.items():
            if filename in item[1]["file_name"]:
                return self.get_image_by_id(item[1]["id"])

if __name__ == "__main__":
    coco = COCO("./captions_val2014.json")

    # print(coco.find_image("000000050403"))
    # print(coco.find_image("000000537802"))
    print(coco.find_image("000000558012"))
    
    
    # keys = coco.sample_with_topic("man")

    # print(len(keys))
    # print(coco.get_image_by_id(keys[0]))

# def get_coco(dataDir, dataType):
#     # dataDir='/Volumes/T7\Shield'
#     # dataType = "val2014"
#     annFile='%s/annotations/captions_%s.json'%(dataDir,dataType)
#     coco = COCO(annFile)

#     return coco

# def get_image_ids(dataDir, dataType):
#     annFile='%s/annotations/%s.json'%(dataDir,dataType) 

#     with open(annFile, "r") as f:
#         data = json.load(f)

#         image_ids = [im["id"] for im in data["images"]]
    
#     return image_ids

# def get_coco_by_id(coco, imgId):
#     annIds = coco.getAnnIds(imgIds=imgId)
#     anns = coco.loadAnns(annIds)
#     caption = coco.showAnns(anns)

#     img = coco.loadImgs(imgId)[0]

#     return img, caption

