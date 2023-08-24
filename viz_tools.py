import glob
import pandas as pd
# from pyedflib import highlevel
import mne
import matplotlib
import matplotlib.pyplot as plt
# from matplotlib.pyplot import specgram
from matplotlib.pyplot import specgram
import json
import os
from scipy.signal import spectrogram
import numpy as np
from coco_viewer_api_server import COCO
import edf_helper as helper

def viz_image_with_spectogram(target_folder):
    # 428357, '/val2014/COCO_val2014_000000428357.jpg'
    # target_folder = "./raw/attempt_1"
    file = glob.glob(target_folder + "/*.edf")[0]
    csv_key = os.path.split(target_folder)[-1]
    experiement_interval = 6
    hz = 128
    hz_experiement_interval = hz * experiement_interval

    with open("./raw/list.json", "r", encoding="utf-8") as f:
        table = json.load(f)

    raw_data, info, channels = helper.read_edf_file(file)
    print(f"INFO: {info}")
    roi_data, timestamp = helper.process_raw_data(raw_data, channels)

    record_data = table[csv_key]
    df = pd.read_csv("./coco_viewer_api_server/" + record_data)
    json_df = df.to_dict(orient='records')

    channels = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']

    import random
    rows = random.sample(json_df, 2)

    for new_row in rows:
    #    id,src,caption,start,end,width,height
        out = "./dataset/{id}_c_{channel}.png"
        start = float(f"{(new_row['start']/1000):.3f}")
        end = start + experiement_interval
        if end > new_row["end"]: continue

        target_edf_data = helper.get_roi_content(roi_data, timestamp, start, end)
        new_row["spectogram"] = []

        if target_edf_data.shape[1] > hz_experiement_interval:
            target_edf_data = target_edf_data[:, :hz_experiement_interval]
        length = target_edf_data.shape[1]
        target_edf_data = np.pad(target_edf_data, (0, hz_experiement_interval-length), mode="mean")

        for i in range(target_edf_data.shape[0]):
            out_name = out.format(id=new_row['id'], channel=i)
            helper.draw_spectogram(target_edf_data[i, :], out_name)
            new_row["spectogram"].append(out_name)

        print(new_row)
