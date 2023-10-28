import glob
import pandas as pd
import mne
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram
import json
import os
from scipy.signal import spectrogram
import numpy as np

# READ the raw edf file 
# return the data and info and edf channels
def read_edf_file(file):
    data = mne.io.read_raw_edf(file)
    raw_data = data.get_data()
    info = data.info    
    channels = data.ch_names

    return raw_data, info, channels


# Process the raw data into roi data
# return timestamp for targetting specific time barrier
def process_raw_data(raw_data, channels):
    s_header = "TIME_STAMP_s"
    ms_header = "TIME_STAMP_ms"
    s_channel_idx = channels.index(s_header)
    ms_channel_idx = channels.index(ms_header)

    roi_channels = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
    roi_channel_idxes = [channels.index(c) for c in roi_channels]

    timestamp = raw_data[ms_channel_idx, :] * 1000 + raw_data[s_channel_idx, :] * 1000000
    roi_data = raw_data[roi_channel_idxes, :]

    return roi_data, timestamp


# Get a data using roi time area.
def get_roi_content(nd_data, nd_timestamp, start, end, eps=1e-6):
    roi_timestamp = (nd_timestamp>=start)*(nd_timestamp<=end)

    roi_data = nd_data[:, roi_timestamp]
    # roi_data += eps

    return roi_data

def draw_spectrogram(data, out):
    length = data.shape[0]
    # for i in range(length):
    # im = specgram(data, Fs=125, noverlap=1)[3]
    # plt.gca().set_axis_off()
    # plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
    #             hspace = 0, wspace = 0)
    # plt.margins(0,0)
    # plt.gca().xaxis.set_major_locator(matplotlib.ticker.NullLocator())
    # plt.gca().yaxis.set_major_locator(matplotlib.ticker.NullLocator())
    # plt.savefig(out, bbox_inches = 'tight', pad_inches = 0)
    # plt.show()
    raw = spectrogram(data, fs=125, noverlap=1)[2]
    im = specgram(data, Fs=125, noverlap=1)[3]
    plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(matplotlib.ticker.NullLocator())
    plt.gca().yaxis.set_major_locator(matplotlib.ticker.NullLocator())
    plt.savefig(out, bbox_inches = 'tight',
        pad_inches = 0)
    plt.clf()


def process(file, data):
    table = {}

    raw_data, info, channels = read_edf_file(file)
    #print(f"INFO: {info}")
    roi_data, timestamp = process_raw_data(raw_data, channels)

    record_data = table[file]
    df = pd.read_csv(record_data)
    json_df = df.to_dict(orient='records')

    for new_row in json_df:
        out = f"./dataset/{new_row['id']}.png"
        start = new_row["start"]
        end = new_row["start"] + 2
        if end > new_row["end"]: continue

        target_edf_data = get_roi_content(roi_data, timestamp, start, end)
        # draw_mel(target_edf_data, out)
        new_row["spectrogram"] = out
        data.append(new_row)
    
    return data


def build_dataset(path):
    with open("dataset.json", "r", encoding="utf-8") as f:
        try: 
            data = json.load(f)
        except: data = []
 
    files = glob.glob(f"{path}/*.edf")
    for file in files:
        data = process(file, data)

    with open("dataset.json", "w", encoding="utf-8") as f:
        json.dump(data)

if __name__ == "__main__":
    target_folder = "./raw/attempt_1"
    file = glob.glob(target_folder + "/*.edf")[0]
    csv_key = os.path.split(target_folder)[-1]
    experiement_interval = 6
    hz = 128
    hz_experiement_interval = hz * experiement_interval

    with open("./raw/list.json", "r", encoding="utf-8") as f:
        table = json.load(f)

    raw_data, info, channels = read_edf_file(file)
    print(f"INFO: {info}")
    roi_data, timestamp = process_raw_data(raw_data, channels)

    record_data = table[csv_key]
    df = pd.read_csv("./coco_viewer_api_server/" + record_data)
    json_df = df.to_dict(orient='records')

    channels = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']

    import random
    rows = random.sample(json_df, 2)

    for new_row in rows:
        out = "./dataset/{id}_c_{channel}.png"
        start = float(f"{(new_row['start']/1000):.3f}")
        end = start + experiement_interval
        if end > new_row["end"]: continue

        target_edf_data = get_roi_content(roi_data, timestamp, start, end)
        new_row["spectrogram"] = []

        if target_edf_data.shape[1] > hz_experiement_interval:
            target_edf_data = target_edf_data[:, :hz_experiement_interval]
        length = target_edf_data.shape[1]
        target_edf_data = np.pad(target_edf_data, (0, hz_experiement_interval-length), mode="mean")

        for i in range(target_edf_data.shape[0]):
            out_name = out.format(id=new_row['id'], channel=i)
            draw_spectrogram(target_edf_data[i, :], out_name)
            new_row["spectrogram"].append(out_name)

        print(new_row)