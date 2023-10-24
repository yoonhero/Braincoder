from pathlib import Path
from PIL import Image

# 7506
imgs = ["COCO_val2014_000000558012.jpg", "COCO_val2014_000000007506.jpg", "COCO_val2014_000000388569.jpg"]

base = Path("/Volumes/T7/coco_viewer_web/static/val2014/")

for img in imgs:
    dir = str(base / img)
    Image.open(dir).save(img)