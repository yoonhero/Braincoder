from PIL import Image
import pandas as pd
import os
from pathlib import Path
import hashlib

BLACK = (0,0,0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
CYAN = (0, 255, 255)
MAGENTA = (255, 0, 255)

COLORS = {"black": BLACK, "white": WHITE, "red": RED, "green": GREEN, "blue": BLUE, "yellow": YELLOW, "cyan": CYAN, "magenta": MAGENTA}

os.makedirs("colors", exist_ok=True)

files = {}
for name, rgb in COLORS.items():
    img = Image.new('RGB', (900, 600), rgb)
    source_dir = f"colors/{name}.png"
    # files[name] = os.path.abspath(source_dir)
    files[name] = source_dir
    img.save(source_dir)

my_data = [[hashlib.sha256(str(i).encode()).hexdigest(), n, f, 0] for i, (n, f) in enumerate(files.items())]
img_data = pd.DataFrame(data=my_data, index=range(len(my_data)), columns=["ID","COLOR", "IMG_DIR", "SEEN"])

img_data.to_csv("color_dataset.csv")