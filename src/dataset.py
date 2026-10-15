# src/dataset.py

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class FIWTriadsDataset(Dataset):
    def __init__(self, csv_path: str, embeddings: dict, fiw_root: str,
                 ages_csv_path: str = None, image_size: int = 512,
                 gender_dict: dict = None):  # ← CAMBIO 1
        self.df = pd.read_csv(csv_path)
        self.embeddings = embeddings
        self.fiw_root = fiw_root
        self.gender_dict = gender_dict or {}  # ← CAMBIO 2

        if ages_csv_path and os.path.exists(ages_csv_path):
            ages_df = pd.read_csv(ages_csv_path)
            self.ages = dict(zip(ages_df["key"], ages_df["age"]))
        else:
            self.ages = {}

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        before = len(self.df)
        self.df = self.df[
            self.df["p1_papa"].isin(embeddings) &
            self.df["p1_mama"].isin(embeddings) &
            self.df["p2"].isin(embeddings)
        ].reset_index(drop=True)
        after = len(self.df)
        if before != after:
            print(f"⚠ Descartadas {before - after} tríadas sin embeddings")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        h_p = torch.from_numpy(self.embeddings[row["p1_papa"]]).float()
        h_m = torch.from_numpy(self.embeddings[row["p1_mama"]]).float()
        h_c = torch.from_numpy(self.embeddings[row["p2"]]).float()

        img_path = os.path.join(self.fiw_root, row["p2"])
        img = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(img)

        age = max(1, int(self.ages.get(row["p2"], 25)))

        # ← CAMBIO 3: parsear key del hijo para obtener género
        parts = row["p2"].split("/")
        gender_key = f"{parts[0]}/{parts[1]}"
        gender = self.gender_dict.get(gender_key, "person")

        return {
            "h_p": h_p, "h_m": h_m, "h_c": h_c,
            "img": img_tensor,
            "age": age,
            "gender": gender,
            "key_p": row["p1_papa"],
            "key_m": row["p1_mama"],
            "key_c": row["p2"],
        }