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
from utils import load_json, write_json

@lru_cache(maxsize=None)
def _get():
    all = glob.glob("./dataset/*")
    return [dir.split("\\")[-1] for dir in all]
total = _get()

def is_exist(outname):    
    out = outname.split("/")[-1]
    return out in total


def _draw(data):
    target, out_name = data
    exist = is_exist(out_name)

    if not exist:
        start_time_sec = time.time()
        draw_spectrogram(target, out_name)
        print(f"time spend {time.time() - start_time_sec}")
        # row["spectrogram"].append(out_name)
        # print(row)
    return out_name


def get_coco_in_same_directory(file):
    coco_parent_path = Path(file).parent.__str__() 
    coco_data_path = glob.glob(f"{coco_parent_path}/coco*.csv")[0]
    return coco_data_path


def build_dataset_from_one_exp(target_folder, parallel):
    file = glob.glob(target_folder + "/*.edf")[0]
    #csv_key = os.path.split(target_folder)[-1]
    experiement_interval = 2
    hz = 128
    hz_experiement_interval = hz * experiement_interval

    # READ EDF FILE AND PROCESSING FOR FURTHER DRAW SPECTO
    raw_data, info, channels = read_edf_file(file)
    #print(f"INFO: {info}")
    roi_data, timestamp = process_raw_data(raw_data, channels)

    # GET RECORD DATA
    record_data_path = get_coco_in_same_directory(file)
    # record_data = tabl[csv_key]
    # df = pd.read_csv(f"./raw/{csv_key}/{record_data}")
    df = pd.read_csv(record_data_path)
    json_df = df.to_dict(orient='records')

    # ROI CHANNELS FOR SELECTING
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

            # USING MULTIPROCESSING AS SPECTOGRAM DRAWING PARALLELISM (Spending 0.6~1.2s*14*3500)
            # out_name = out.format(id=row['id'], channel=i)
            out_names = [out.format(id=row["id"], channel=i) for i in range(total_channels)]
            targets = [target_edf_data[i, :] for i in range(total_channels)]
            #ok = [is_exist(n) for n in out_names]

            # datas = [(a, b, c) for a, b, c in zip(targets, out_names, ok)]
            datas = [(a, b) for a, b in zip(targets, out_names)]

            if parallel:
                with multiprocessing.Pool(os.cpu_count() - 1) as p:
                    res = p.map(_draw, datas)
                    row["spectrogram"] = res
            else:
                res = []
                for data in datas:
                    _res = _draw(data)
                    res.append(_res)
                row["spectogram"] = res

        except: print(f"WARNING WHAT'S WRONG FFFF, {row['id']} {target_folder}")

        # for i __annotations__in range(target_edf_data.shape[0]):
        dataset.append(row)
    
    return dataset


def build_dataset(excludes=[], parallel=False):
    temp_dests = glob.glob("./raw/*")

    dataset = []

    for dest in temp_dests:
        if ".json" not in dest and dest.split("/")[-1] not in excludes:
            temp_dataset = build_dataset_from_one_exp(dest, parallel)

            dataset = dataset + temp_dataset
            print(f"finished processing!! {dest}")

    # with open("dataset.json", "w", encoding="utf-8") as f:
    #     json.dump(dataset, f)
    write_json("./dataset.json", dataset)


def clean_dataset_and_select_eval_items(exclude):
    dataset = load_json("./dataset.json")

    train = list(filter(lambda x: x["id"] not in exclude, dataset))
    write_json("./train_dataset.json", train)
    print(f"Train Dataset Size: {len(train)} samples!!")

    eval = list(filter(lambda x: x["id"] in exclude, dataset))
    return eval

def selecting_keys_for_eval(participant_name):
    exps = glob.glob(f"./raw/*/*{participant_name}*.csv")
    
    interest_csv_path = [get_coco_in_same_directory(exp) for exp in exps]

    keys = []
    for csv_path in interest_csv_path:
        df = pd.read_csv(csv_path)
        temp_keys = df.loc[:, "id"].tolist()
        keys = keys + temp_keys
    
    return keys

# Bulidng evaluation dataset because of 
def build_eval(participants=[]):
    keys = []
    for participant in participants:
        temp_keys = selecting_keys_for_eval(participant_name=participant)
        keys = keys + temp_keys

    eval_data = clean_dataset_and_select_eval_items(exclude=keys)
    write_json("./eval_dataset.json", eval_data)
    print(f"Eval Dataset Size: {len(eval_data)} samples!!")

def ohmygoooooooooooooooooooooooooooood():
    ffff = ["./raw/attempt_8_3_lhb", "./raw/attempt_8_3_real_lsh", "./raw/attempt_8_4_lsh"]

    for dest in ffff:
        temp_dataset = build_dataset_from_one_exp(dest, True)
        print(f"finished processing!! {dest}") 

if __name__ == "__main__":
    # build_dataset(parallel=False)

    eval_participants = ["zsh"]
    build_eval(eval_participants)

    # ohmygoooooooooooooooooooooooooooood()