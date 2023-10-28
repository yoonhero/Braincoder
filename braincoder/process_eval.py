import numpy as np
import tqdm
import multiprocessing
import os
import glob

from edf_helper import read_edf_file, process_raw_data, get_roi_content, draw_spectrogram


def get_info_table(participant):
    txt_path = f"./eval/datas/{participant}.txt"
    with open(txt_path, "r", encoding="utf-8") as f:
        content = f.read()

    time_data = []
    rows = [row for row in content.split("-")]
    for item in rows[1:]:
        start = int(item.split(":")[-1])
    
        time_data.append(start)

    table = {}
    to = list(zip(time_data, time_data[1:]))+[(time_data[-1], time_data[-1]+10)]
    print(to)
    for i, (start, end) in enumerate(to):
        # START
        one = (start+0.5, start+2.5)

        # MIDDLE
        halfdis = (end-start)/2
        two = (start+halfdis-1, start+halfdis+1)

        # END
        three = (end-2.5, end-.5)

        _d = [one, two, three]
        table[i] = _d

    # table = {1: table[1], 5: table[5], 7: table[7], 9: table[9], }
    return table

def build(participant):
    file = f"./eval/datas/{participant}.edf"
    experiement_interval = 2
    hz = 128
    hz_experiement_interval = hz * experiement_interval

    # READ EDF FILE AND PROCESSING FOR FURTHER DRAW SPECTO
    raw_data, info, channels = read_edf_file(file)
    roi_data, timestamp = process_raw_data(raw_data, channels)

    # GET RECORD DATA
    data = get_info_table(participant)

    # ROI CHANNELS FOR SELECTING
    channels = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']

    total = []

    for row in tqdm.tqdm(list(data.items()), desc=f"Processing {participant}..."):
        index, data = row
        out = "./eval/eeg/{participant}_{id}_{when}_c_{channel}.png"
        whens = ["start", "middle", "end"]

        for when, d in zip(whens,data):
            start, end = d
            expected_end = start + experiement_interval
            if expected_end > end: continue
            target_edf_data = get_roi_content(roi_data, timestamp, start, expected_end)
            if target_edf_data.shape[1] > hz_experiement_interval:
                target_edf_data = target_edf_data[:, :hz_experiement_interval]

            # try:
            length = target_edf_data.shape[1]
            target_edf_data = np.pad(target_edf_data, (0, hz_experiement_interval-length), mode="mean")                
            out_names = [out.format(id=index+1, channel=i, when=when, participant=participant) for i in range(14)]
            datas = [{"data":target_edf_data[i, :], "out":out_names[i]} for i in range(14)]
            total = total + datas
            
            # for aiaiai in range(14):
            #     draw_spectrogram(target_edf_data[aiaiai, :], out_names[aiaiai])

    # with multiprocessing.Pool(os.cpu_count() - 1) as p:
    #     p.map(draw_spectrogram, total)
    already = glob.glob("./eval/eeg/*")
    already = [_dir.split('/')[-1] for _dir in already]
    for a in tqdm.tqdm(total):
        if a["out"].split("/")[-1] in already:
            print("pass")
            continue
        draw_spectrogram(a["data"], a["out"])
    
    return

# experiment_participants = ["jjw", "jsm"]
experiment_participants = ["kdh", "cmj", "jsm", "kjh", "jyh", "kian", "csw", "kho"]
for participant in experiment_participants:
    build(participant)
