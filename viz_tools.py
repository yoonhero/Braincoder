import glob
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import numpy as np
from coco_viewer_api_server.coco import COCO
import edf_helper as helper
from config import experiement_interval, hz, drive_src, channels
import random
import cv2
import tqdm

def viz_subplot_for_spectrogram_compare(data):
    fig = plt.figure()
    gs = fig.add_gridspec(4, 7)
    main = fig.add_subplot(gs[0:2, :])
    main_img_src = drive_src + data["src"]
    img = plt.imread(main_img_src)
    main.imshow(img)
    main.axis("off")
    
    _spec = data["spectrogram"]
    for i in range(14):
        row = i//7+2
        column = i % 7
        ax = fig.add_subplot(gs[row, column])
        img = plt.imread(_spec[i])
        ax.imshow(img)
        ax.set_title(channels[i])
        ax.axis("off")

    fig.suptitle(data["caption"])
    fig.tight_layout()
    out = f'{data["id"]}.png'
    fig.savefig(out, dpi=400)

def viz_image_with_spectrogram(target_folder):
    # 428357, '/val2014/COCO_val2014_000000428357.jpg'
    # target_folder = "./raw/attempt_1"
    file = glob.glob(target_folder + "/*.edf")[0]
    csv_key = os.path.split(target_folder)[-1]
    hz_experiement_interval = hz * experiement_interval

    with open("./raw/list.json", "r", encoding="utf-8") as f:
        table = json.load(f)

    raw_data, info, channels = helper.read_edf_file(file)
    print(f"INFO: {info}")
    roi_data, timestamp = helper.process_raw_data(raw_data, channels)

    record_data = table[csv_key]
    df = pd.read_csv("./coco_viewer_api_server/" + record_data)
    json_df = df.to_dict(orient='records')

    random_rows = random.sample(json_df, 4)

    datas = []
    for new_row in random_rows:
        out_format = "./dataset/{id}_c_{channel}.png"
        start = float(f"{(new_row['start']/1000):.3f}")
        end = start + experiement_interval
        if end > new_row["end"]: continue

        target_edf_data = helper.get_roi_content(roi_data, timestamp, start, end)
        new_row["spectrogram"] = []

        if target_edf_data.shape[1] > hz_experiement_interval:
            target_edf_data = target_edf_data[:, :hz_experiement_interval]
        length = target_edf_data.shape[1]
        target_edf_data = np.pad(target_edf_data, (0, hz_experiement_interval-length), mode="mean")

        for i in tqdm.tqdm(range(target_edf_data.shape[0]), desc="Save EDF Spectrogram channel by channel"):
            out = out_format.format(id=new_row['id'], channel=i)
            helper.draw_spectrogram(target_edf_data[i, :], out)
            new_row["spectrogram"].append(out)

        datas.append(new_row)

    for data in tqdm.tqdm(datas, desc="Drawing Subplots"):
        viz_subplot_for_spectrogram_compare(data)

if __name__ == "__main__":
    viz_image_with_spectrogram("./raw/attempt_1")