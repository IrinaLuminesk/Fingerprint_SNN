import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2
from torch.utils.data.sampler import Sampler
import random
from collections import defaultdict

class SiameseFingerprintDataset(Dataset):
    def __init__(self, image_size, mean, std, batch_size, data_path):
        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.batch_size = batch_size
        self.data_path = data_path 
    def build_dataset(self, data_type):
        dataset = ImageFolder(self.data_path, transform=self.train_transform() if data_type == "Train" else self.test_transform())
        if data_type == "Train":
            self.p = 8
            self.k = 8
            batch_size = self.p * self.k
            self.targets = list(dataset.targets)
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=PKSampler(self.targets, self.p, self.k)
            )
        else:
            loader =  DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        return loader
    def train_transform(self):
        return v2.Compose([
            v2.Resize(self.image_size),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=self.mean,
                std=self.std
            )
        ])
    def test_transform(self):
        return v2.Compose([
            v2.Resize(self.image_size),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=self.mean,
                std=self.std
            )
        ])
    def __len__(self):
        return len(self.targets) // (self.p * self.k)
    
class PKSampler(Sampler):
    def __init__(self, groups, p, k):
        self.p = p
        self.k = k
        self.groups = create_groups(groups, self.k)

        # Ensures there are enough classes to sample from
        if len(self.groups) < p:
            raise ValueError("There are not enough classes to sample from")

    def __iter__(self):
        # Shuffle samples within groups
        for key in self.groups:
            random.shuffle(self.groups[key])

        # Keep track of the number of samples left for each group
        group_samples_remaining = {}
        for key in self.groups:
            group_samples_remaining[key] = len(self.groups[key])

        while len(group_samples_remaining) > self.p:
            # Select p groups at random from valid/remaining groups
            group_ids = list(group_samples_remaining.keys())
            selected_group_idxs = torch.multinomial(torch.ones(len(group_ids)), self.p).tolist()
            for i in selected_group_idxs:
                group_id = group_ids[i]
                group = self.groups[group_id]
                for _ in range(self.k):
                    # No need to pick samples at random since group samples are shuffled
                    sample_idx = len(group) - group_samples_remaining[group_id]
                    yield group[sample_idx]
                    group_samples_remaining[group_id] -= 1

                # Don't sample from group if it has less than k samples remaining
                if group_samples_remaining[group_id] < self.k:
                    group_samples_remaining.pop(group_id)

def create_groups(groups, k):
    group_samples = defaultdict(list)
    for sample_idx, group_idx in enumerate(groups):
        group_samples[group_idx].append(sample_idx)

    keys_to_remove = []
    for key in group_samples:
        if len(group_samples[key]) < k:
            keys_to_remove.append(key)
            continue

    for key in keys_to_remove:
        group_samples.pop(key)

    return group_samples