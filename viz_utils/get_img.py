from pathlib import Path
from PIL import Image

# 7506
# imgs = ["COCO_val2014_000000558012.jpg", "COCO_val2014_000000007506.jpg", "COCO_val2014_000000388569.jpg", "COCO_val2014_000000076525.jpg"]
# imgs = ["COCO_val2014_000000076525.jpg", "COCO_val2014_000000561810.jpg", "COCO_val2014_000000273878.jpg", "COCO_val2014_000000483050.jpg", "COCO_val2014_000000301765.jpg"]
imgs = ["COCO_val2014_000000006471.jpg"]

# 21613, 55772, 173685, 20820, 332877, 30254, 264853, 28194
# 333924_c_0, 351597_c_0, 465715_c_0, 540006_c_0, 542248_c_0
imgs = ["COCO_val2014_000000021613.jpg", "COCO_val2014_000000055772.jpg", "COCO_val2014_000000173685.jpg", "COCO_val2014_000000020820.jpg","COCO_val2014_000000332877.jpg", "COCO_val2014_000000030254.jpg", "COCO_val2014_000000264853.jpg", "COCO_val2014_000000028194.jpg"]
imgs = ["COCO_val2014_000000169152.jpg"]
imgs = ["COCO_val2014_000000333924.jpg", "COCO_val2014_000000351597.jpg", "COCO_val2014_000000465715.jpg", "COCO_val2014_000000540006.jpg","COCO_val2014_000000542248.jpg"]

base = Path("/Volumes/T7/coco_viewer_web/static/val2014/")

for img in imgs:
    dir = str(base / img)
    Image.open(dir).save(img)