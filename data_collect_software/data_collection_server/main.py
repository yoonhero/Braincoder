import time
start = time.time()*1000
print(start)

from flask import Flask, jsonify, make_response, request
from flask_cors import CORS, cross_origin
import pandas as pd
import random
from PIL import Image
import base64
from io import BytesIO
import logging
import os
from datetime import datetime
import time
import uuid
from functools import lru_cache
import glob
import json
from coco import COCO

app = Flask(__name__)
cors = CORS(app, resources={r"*": {"origins": "*"}})

file_des = f"coco_{int(time.time())}.csv"
# df = pd.read_csv(file_des, index_col=0)
index = ["id", "src", "caption", "start", "end", "width", "height"]
df = pd.DataFrame([], columns=index)

def create_uid():
    return str(uuid.uuid1())

def update_df():
    df.to_csv(file_des)

keywords = ['Dog', 'Cat', 'Elephant', 'Giraffe', 'Lion', 'Tiger', 'Bear', 'Dolphin', 'Penguin', 'Monkey', 'Gorilla', 'Kangaroo', 'Zebra', 'Cheetah', 'Rhinoceros', 'Hippopotamus', 'Crocodile', 'Snake', 'Eagle', 'Owl', 'Dolphin', 'Shark', 'Octopus', 'Penguin', 'Panda', 'Koala', 'Fox', 'Wolf', 'Horse', 'Cow', 'Sheep', 'Goat', 'Chicken', 'Pig', 'Rabbit', 'Hamster', 'Guinea Pig', 'Parrot', 'Peacock', 'Flamingo', 'Butterfly', 'Bee', 'Ladybug', 'Ant', 'Spider', 'Scorpion', 'Jellyfish', 'Starfish', 'Sea Turtle', 'Lobster']
keywords = ["landscape", "rock", "sea", "sky"]

coco = COCO("./captions_val2014.json")
image_ids = list(coco.get_image_ids())
# image_ids = coco.sample_with_topic(keywords)
base_dir = "/val2014/"

def load_seen():
    seen = []
    try:
        with open("seen.json", "r") as f:
            seen = json.load(f)["seen"]
    except:
        pass
    return seen

##### SEEN Data
seen = load_seen()

def save_seen():
    with open("seen.json", "w") as f:
        json.dump({"seen": seen}, f)

def sampling():
    while True:
        im, captions = coco.sampling(image_ids)
        caption = captions[0]["caption"]
        if im["id"] not in seen and im["width"] > 400:
            break
    
    return im, caption

@app.route('/getimg', methods=["GET"])
@cross_origin()
def board():
    global df
    image_data, caption = sampling()
    image_id = image_data["id"]

    width, height = image_data["width"], image_data["height"]
    max_height = 600
    width = width * (max_height/height)

    imgdir = base_dir + image_data["file_name"]

    new_row = pd.DataFrame([[image_id, imgdir, caption, 0, 0, width, max_height]], columns=index)
    df = pd.concat([df, new_row])
    
    data = {"url": imgdir, "width": width, "height": max_height, "id": image_id}
    return jsonify(data)

@app.route("/see", methods=["POST"])
@cross_origin()
def see():
    global df, start
    data = request.json
    if data["id"] == "":
        return jsonify({"error": "oh no"})
    
    end = time.time() * 1000
    temp_start = end - 6000

    df.loc[df["id"] == data["id"], "end"] = end - start
    df.loc[df["id"] == data["id"], "start"] = temp_start - start

    update_df()

    seen.append(data["id"])
    save_seen()

    return jsonify({"error": False})

if __name__ == "__main__":
    app.run(host="localhost",port=8000)
