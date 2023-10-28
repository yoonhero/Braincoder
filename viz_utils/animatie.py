import glob
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from pygifmaker.pygifmaker import GifMaker

gifmaker = GifMaker()

files = glob.glob("./gradual_animation/*")
files = sorted(files, key=lambda x: int(x.split("_")[-2]))

imgs = [Image.open(file) for file in files]

myFont = ImageFont.truetype('MaruBuri-Bold.ttf', 35)

for i, im in enumerate(imgs):
    I1 = ImageDraw.Draw(im)
    # Add Text to an image
    I1.text((28, 36), f"Epoch: {i+1}", font=myFont, fill=(0, 0, 0))
    imgs[i] = im

# gifmaker.PIL(imgs, 4, 0)
GifMaker.PIL("output.gif", imgs, 4, 0)
