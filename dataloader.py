import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms as T

import pandas as pd
import json
import time
from PIL import Image
from functools import lru_cache
import multiprocessing
import os
import h5py
from argparse import ArgumentParser

from diffusion_helper import prepare_text_embedding, text2emb
from utils import load_json, load_spectos


def _sort_key(x):
    return int(x.split("_")[-1].split(".")[0])


class COCOCOCOCOCCOCOOCOCOCOCCOCOCOCODatset(Dataset):
    def __init__(self, dataset_path, image_dir, device, width=None, height=None, from_cache=False, cache_dir=None, caching=False):
        self.device = device
        self.width = width
        self.height = height 
        self.from_cache = from_cache
        self.image_dir = image_dir

        if width != None and height != None:
            self.transforms = T.Compose([
                T.Resize((width, height)),
                T.ToTensor()
            ])
        else:
            self.transforms = T.Compose([
                T.ToTensor()
            ])

        start_time_sec = time.time()
        
        # id,src,caption,start,end,width,height, spectogram
        _data = load_json(dataset_path)
        dataset = []
        for d in _data:
            try:
                _spec = d["spectogram"]
            except: 
                print(f"WARNING: this data has problem {d}")
                continue

            if len(_spec) != 14:
                print(f"WARNING: this data has problem {d}")
                continue
            
            _sorted_spec = sorted(_spec, key=_sort_key)
            # change image directory for loading in various environment.
            _cleaned_spec = list(map(lambda x: f"{image_dir}/{x.split('/')[-1]}", _sorted_spec))
            data_row = (d["id"], _cleaned_spec, d["caption"])

            dataset.append(data_row)
        self.dataset = dataset

        self.tokenizer, self.text_encoder = prepare_text_embedding(device=device)
        if caching:
            torch.multiprocessing.set_start_method('spawn')
            self.save_caption_emb()
        elif self.from_cache:
            self.load_cache(cache_dir)

        print(f"Finishing Loading Dataset in {time.time() - start_time_sec}s")
    
    def __getitem__(self, index):
        im_id, specto, caption = self.dataset[index]

        x = load_spectos(specto, self.transforms, self.device)

        if not self.from_cache:
            _, y = text2emb(caption, self.tokenizer, self.text_encoder, self.device)
        else:
            y = self.get_emb_from_cache(im_id)
            y = torch.from_numpy(y[:]).squeeze(0).to(self.device)

        return x, y, im_id

    def save_caption_emb(self):
        # with multiprocessing.Pool(os.cpu_count() - 1) as p:
            # result = p.map(self.to_emb, self.dataset)

        result = []
        for item in self.dataset:
            temp = self.to_emb(item)
            result.append(temp)

        f = h5py.File("cache.hdf5", "w")

        # f.create_dataset()
        for (id, embedding) in result:
            doc_id = f"data/{id}"
            f.create_dataset(doc_id, data=embedding)
        
        f.close()
    
    def load_cache(self, cache_dir):
        f = h5py.File(cache_dir, "r")
        self.cache_dataset = f["data"]
    
    def defer(self):
        self.cache_dataset.close()
    
    def get_emb_from_cache(self, id):
        return self.cache_dataset[str(id)]

    def to_emb(self, data):
        _, embedding = text2emb(data[2], self.tokenizer, self.text_encoder, self.device)
        np_emb = embedding.cpu().numpy()
        return (data[0], np_emb)
        
    def __len__(self):
        return len(self.dataset)


def create_dataloader(batch_size, image_dir, cache_dir, device, seed=1234):
    G = torch.Generator()
    G.manual_seed(seed)
    train_dataset = COCOCOCOCOCCOCOOCOCOCOCCOCOCOCODatset("./train_dataset.json", image_dir, device=device, width=320, height=240, from_cache=True, cache_dir=cache_dir)
    eval_dataset = COCOCOCOCOCCOCOOCOCOCOCCOCOCOCODatset("./eval_dataset.json", image_dir, device=device, width=320, height=240, from_cache=True, cache_dir=cache_dir)

    torch.multiprocessing.set_start_method('spawn')
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=G, num_workers=1)
    evalloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, generator=G, num_workers=1)

    return trainloader, evalloader


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--path", type=str)
    args = parser.parse_args()
    path = args.path

    # PreCaching for memory efficient training pipeline.
    dataset = COCOCOCOCOCCOCOOCOCOCOCCOCOCOCODatset(path, "cuda", width=320, height=240, caching=True)

    for i in range(len(dataset)):
        x, y = dataset[i]

