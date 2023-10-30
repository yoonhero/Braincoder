import numpy as np
import tqdm
import multiprocessing
import os
import glob

from edf_helper import read_edf_file, process_raw_data, get_roi_content, draw_spectrogram


def build(area):
    file = f"./eval/sungho_data/lsh_{area}.edf"
    experiement_interval = 2
    hz = 128
    hz_experiement_interval = hz * experiement_interval

    # READ EDF FILE AND PROCESSING FOR FURTHER DRAW SPECTO
    raw_data, info, channels = read_edf_file(file)
    roi_data, timestamp = process_raw_data(raw_data, channels)

    # GET RECORD DATA
    area2interval = {"game":(25, 45), "ski":(20, 40), "soccer":(110, 130)}    
    _start, _end = area2interval[area] 
    interval = 0.5
    data = [(_start+interval*i, _start+interval*i+experiement_interval) for i in range(int((_end-_start-2)/interval))]

    # ROI CHANNELS FOR SELECTING
    channels = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']

    total = []

    for im, row in enumerate(tqdm.tqdm(data, desc=f"Processing {area}...")):
        start, end = row
        out = "./eval/sungho/{area}_{im}_c_{channel}.png"

        expected_end = start + experiement_interval
        if expected_end > end: continue
        target_edf_data = get_roi_content(roi_data, timestamp, start, expected_end)
        if target_edf_data.shape[1] > hz_experiement_interval:
            target_edf_data = target_edf_data[:, :hz_experiement_interval]

        # try:
        length = target_edf_data.shape[1]
        target_edf_data = np.pad(target_edf_data, (0, hz_experiement_interval-length), mode="mean")                
        out_names = [out.format(channel=i, area=area, im=im) for i in range(14)]
        datas = [{"data":target_edf_data[i, :], "out":out_names[i]} for i in range(14)]
        total = total + datas

    already = glob.glob("./eval/sungho_data/*")
    already = [_dir.split('/')[-1] for _dir in already]
    for a in tqdm.tqdm(total):
        if a["out"].split("/")[-1] in already:
            print("pass")
            continue
        draw_spectrogram(a["data"], a["out"])
    
    return

areas = ["game", "ski", "soccer"]
for area in areas:
    build(area)