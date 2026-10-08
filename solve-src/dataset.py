import os
import json
from typing import List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from vocab import encode_tokens, value_to_token, BAR_TOKEN


def values_to_tokens(values: List[int]) -> List[str]:
    tokens = []
    for i, v in enumerate(values):
        tokens.append(value_to_token(v))
        if (i + 1) % 8 == 0 and (i + 1) < len(values):
            tokens.append(BAR_TOKEN)
    return tokens


class ScoreDataset(Dataset):
    def __init__(self, image_dir: str, labels_path: str):
        self.image_dir = image_dir
        self.rows = []

        with open(labels_path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                tokens = values_to_tokens(row["values"])
                row["tokens"] = tokens
                row["encoded_tokens"] = encode_tokens(tokens)
                self.rows.append(row)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        img_path = os.path.join(self.image_dir, row["image"])

        img = Image.open(img_path).convert("L")
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = 1.0 - arr

        x = torch.from_numpy(arr).unsqueeze(0)
        y = torch.tensor(row["encoded_tokens"], dtype=torch.long)
        return x, y, row["tokens"]


def collate_fn(batch):
    xs, ys, raw = zip(*batch)
    xs = torch.stack(xs, dim=0)
    target_lengths = torch.tensor([len(y) for y in ys], dtype=torch.long)
    targets = torch.cat(ys, dim=0)
    return xs, targets, target_lengths, raw
