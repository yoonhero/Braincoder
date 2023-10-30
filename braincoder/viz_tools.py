import glob
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import numpy as np
import edf_helper as helper
from config import experiement_interval, hz, drive_src, channels
import random
# import cv2
import tqdm
from sklearn.manifold import TSNE
import seaborn as sns

import sys
sys.path.append('../viz_utils')
from coco import COCO


def viz_subplot_for_spectrogram_compare(data):
    fig = plt.figure()
    gs = fig.add_gridspec(4, 7)
    main = fig.add_subplot(gs[0:2, :])
    # main_img_src = drive_src + data["src"]
    main_img_src = data["src"]
    img = plt.imread(main_img_src, format="jpg")
    main.imshow(img)
    main.axis("off")
    
    _spec = data["spectrogram"]
    for i in range(14):
        row = i//7+2
        column = i % 7
        ax = fig.add_subplot(gs[row, column])
        if not isinstance(_spec[i], np.ndarray):
            img = plt.imread(_spec[i])
        else: img = _spec[i]
        ax.imshow(img)
        ax.set_title(channels[i])
        ax.axis("off")

    fig.suptitle(data["caption"])
    fig.tight_layout()
    out = f'{data["id"]}.png'
    # dot per inch
    fig.savefig(out, dpi=400)


def viz_image_with_spectrogram(target_folder):
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

def show_tsne(x, y, title):
    tsne = TSNE(n_components=2, verbose=1, random_state=123)
    z = tsne.fit_transform(x)
    df = pd.DataFrame()
    df["y"] = y
    df["feature-1"] = z[:, 0]
    df["feature-2"] = z[:, 1]

    sns.scatterplot(x="feature-1", y="feature-2", hue=df.y.tolist(), palette=sns.color_palette("hls", 3),
                data=df).set(title=title)

def tsne_specific_channel(dir, channel_idx):
    files = glob.glob(dir + f"/*{channel_idx}.png")
    title = f"Channel {channels[channel_idx]} Gazing Data Difference"

    coco = COCO("./coco_viewer_api_server/captions_val2014.json")

    for file in files:
        img_id = os.path.split(file)[-1].split("_")[0]
        print(img_id)
        image, ann = coco.get_image_by_id(img_id)
        print(ann)
        break


if __name__ == "__main__":
    # viz_image_with_spectrogram("./raw/attempt_1")
    # tsne_specific_channel("./dataset", 0)
    # ({'license': 1, 'url': 'http://farm4.staticflickr.com/3228/2755941377_ea852330ca_z.jpg', 'file_name': 'COCO_val2014_000000006471.jpg', 'id': 6471, 'width': 500, 'date_captured': '2013-11-19 18:09:09', 'height': 333}, [{'image_id': 6471, 'id': 735287, 'caption': 'A baseball player holding a bat next to a base.'}, {'image_id': 6471, 'id': 738365, 'caption': 'A hitter is waiting for the pitch to be thrown'}, {'image_id': 6471, 'id': 743219, 'caption': 'A baseball player in a batting stance at home plate with the catcher in position to catch a pitch.'}, {'image_id': 6471, 'id': 827020, 'caption': 'A baseball player holds his bat and waits for the pitch. '}, {'image_id': 6471, 'id': 828315, 'caption': 'A man at bat waiting for a pitch with a catcher and umpire behind him and players from the opposing team in the dugout.'}])
    spectrograms = [f"/Users/yoonseonghyeon/Downloads/dataset/6471_c_{c}.png" for c in range(14)]
    data = {"spectrogram": spectrograms, "src": "http://farm4.staticflickr.com/3228/2755941377_ea852330ca_z.jpg", "caption": "A baseball player holding a bat next to a base.", "id": 6471}
    viz_subplot_for_spectrogram_compare(data)