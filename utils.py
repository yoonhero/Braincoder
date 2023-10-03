import pandas as pd
import json
from functools import lru_cache
from PIL import Image
import torch


def make_indexing(path):
    df = pd.read_csv(path)

    temp_data = df.loc[:, ["id", "src", "caption"]]
    ids = temp_data["id"].to_list()
    temp_data = temp_data.set_index("id")
    dict_data = temp_data.to_dict()
    
    result = {_id:{"src":dict_data["src"][_id], "caption":dict_data["caption"][_id]} for _id in ids}

    return result

def make_index_table(paths):
    result = {}

    for path in paths: result.update(make_indexing(path))

    return result

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)

    return d

def write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def get_image_info_by_id(table_path, id):
    @lru_cache
    def _load(table_path):
        return load_json(table_path)
    index_table = _load(table_path)

    return index_table.get(id)


def load_spectos(paths, transform, device):
    images = [Image.open(path).convert("RGB") for path in paths]
    transformed = [transform(im) for im in images]

    stacking = torch.cat(transformed, dim=0).to(device)
    stacking /= 255
    return stacking