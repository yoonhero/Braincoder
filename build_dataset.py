import glob
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
import tqdm
import multiprocessing
import time
from functools import lru_cache

from edf_helper import read_edf_file, process_raw_data, get_roi_content, draw_spectrogram


def _draw(data):
    target, out_name, exist = data
    if not exist:
        start_time_sec = time.time()
        draw_spectrogram(target, out_name)
        print(f"time spend {time.time() - start_time_sec}")
        # row["spectrogram"].append(out_name)
        # print(row)
    return out_name


total = glob.glob("./dataset/*")
def is_exist(outname):    
    for t in total:
        if outname in t:
            return True 

    return False


def build_dataset_from_one_exp(target_folder):
    file = glob.glob(target_folder + "/*.edf")[0]
    csv_key = os.path.split(target_folder)[-1]
    experiement_interval = 2
    hz = 128
    hz_experiement_interval = hz * experiement_interval

    # with open("./raw/list.json", "r", encoding="utf-8") as f:
    #     table = json.load(f)

    raw_data, info, channels = read_edf_file(file)
    print(f"INFO: {info}")
    roi_data, timestamp = process_raw_data(raw_data, channels)

    record_data_parent_path = Path(file).parent.__str__() 
    record_data_path = glob.glob(f"{record_data_parent_path}/coco*.csv")[0]
    # record_data = tabl[csv_key]
    # df = pd.read_csv(f"./raw/{csv_key}/{record_data}")
    df = pd.read_csv(record_data_path)
    json_df = df.to_dict(orient='records')

    channels = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']

    dataset = []
    temp = 0
    for row in tqdm.tqdm(json_df, desc=f"Processing {target_folder}..."):
    #    id,src,caption,start,end,width,height
        out = "./dataset/{id}_c_{channel}.png"
        start = float(f"{(row['start']/1000):.3f}")
        expected_end = start + experiement_interval
        if expected_end > row["end"]/1000: continue
        if temp > row["end"]:   
            print(f"WARNING!! Problem in {target_folder} {row}")
            break
        temp = row["end"]

        target_edf_data = get_roi_content(roi_data, timestamp, start, expected_end)
        row["spectrogram"] = []
        
        if target_edf_data.shape[1] > hz_experiement_interval:
            target_edf_data = target_edf_data[:, :hz_experiement_interval]

        try:
            length = target_edf_data.shape[1]
            total_channels =  target_edf_data.shape[0]
            target_edf_data = np.pad(target_edf_data, (0, hz_experiement_interval-length), mode="mean")

            with multiprocessing.Pool(os.cpu_count() - 1) as p:
                # out_name = out.format(id=row['id'], channel=i)
                out_names = [out.format(id=row["id"], channel=i) for i in range(total_channels)]
                targets = [target_edf_data[i, :] for i in range(total_channels)]
                ok = [is_exist(n) for n in out_names]

                datas = [(a, b, c) for a, b, c in zip(targets, out_names, ok)]

                res = p.map(_draw, datas)
                # res = _draw(datas[0])
                print(res)
                row["spectrogram"] = res
        except: print(f"WARNING WHAT'S WRONG FFFF, {row['id']} {target_folder}")

        # for i in range(target_edf_data.shape[0]):
        dataset.append(row)
    
    return dataset


def build_dataset(excludes=[]):
    temp_dests = glob.glob("./raw/*")

    dataset = []

    for dest in temp_dests:
        if ".json" not in dest and dest.split("/")[-1] not in excludes:
            temp_dataset = build_dataset_from_one_exp(dest)

            dataset = dataset + temp_dataset
            print(f"finished processing!! {dest}")

    with open("dataset.json", "a", encoding="utf-8") as f:
        json.dump(dataset, f)
    

if __name__ == "__main__":
    build_dataset()