import torch
from torch.utils.data import Dataset
import numpy as np

class EntityMatchDataset(Dataset):
    def __init__(self, records_a, records_b, features_list, labels, tokenizer, max_length=256):
        """
        records_a: list of tuples (entity_id, name, address, country)
        records_b: list of tuples (entity_id, name, address, country)
        features_list: list of numeric feature vectors from features.py
        labels: list of 1s and 0s
        """
        self.records_a = records_a
        self.records_b = records_b
        self.features_list = features_list
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        a = self.records_a[idx]
        b = self.records_b[idx]
        features = self.features_list[idx]
        label = self.labels[idx]

        # Format: Name Address Country
        text_a = f"{a[1]} {a[2]} {a[3]}"
        text_b = f"{b[1]} {b[2]} {b[3]}"

        encoded = self.tokenizer(
            text_a,
            text_b,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "features": torch.tensor(features, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.float32)
        }
