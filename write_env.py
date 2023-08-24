import sys

link = sys.argv[-1]
print(link)

with open("/Volumes/T7/coco_viewer_web/.env", "w") as f:
    f.write(f"REACT_APP_API_LINK = {link}")
