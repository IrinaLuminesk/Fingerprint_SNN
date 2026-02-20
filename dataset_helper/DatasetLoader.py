from turtle import st
from torchvision.transforms import v2
from torchvision import tv_tensors
from torch.utils.data import DataLoader, Dataset
import torch

from pathlib import Path
from collections import defaultdict
import os
import random
from PIL import Image

class SiameseFingerprintDataset(Dataset):
    def __init__(self, path, N, image_size, mean, std, transform_type="train", enabled_transform=False):
        self.path = Path(path) #Đường dẫn dataset
        self.N = N #Số lượng pair sẽ được tạo
        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.transform_type = transform_type
        self.enabled_transform = enabled_transform
        #Tạo contraint
        self.person_ids = [i for i in os.listdir(self.path)]
        if N < 2 * len(self.person_ids):
            raise ValueError("N={0} is too small. Need at least {1} pairs so every person is used at least once.".format(
                self.N, 2 * len(self.person_ids)
            ))

        self.person_to_images = self.Create_Person_Images()

        self.pairs = self.Create_Pair()
    def Create_Person_Images(self):
        person_fingerprints = defaultdict(list)
        for i in self.path.rglob("*"):
            if i.is_file():
                person_fingerprints[str(i.name).split("_")[0]].append(str(i))
        return person_fingerprints
    def Create_Pair(self):
        half = self.N // 2

        # -------- Genuine pairs (label = 1) --------
        genuine_pairs = []

        # ensure each person appears at least once
        for person in self.person_ids:
            imgs = self.person_to_images[person]
            img1, img2 = random.sample(imgs, 2)
            genuine_pairs.append((img1, img2, 1))

        # fill remaining genuine pairs
        while len(genuine_pairs) < half:
            person = random.choice(self.person_ids)
            imgs = self.person_to_images[person]
            img1, img2 = random.sample(imgs, 2)
            genuine_pairs.append((img1, img2, 1))

        # -------- Imposter pairs (label = 0) --------
        imposter_pairs = []

        while len(imposter_pairs) < half:
            p1, p2 = random.sample(self.person_ids, 2)
            img1 = random.choice(self.person_to_images[p1])
            img2 = random.choice(self.person_to_images[p2])
            imposter_pairs.append((img1, img2, 0))

        pairs = genuine_pairs[:half] + imposter_pairs
        random.shuffle(pairs)
        return pairs 
    def train_transform(self):
        if self.enabled_transform == False:
            return v2.Compose([
                v2.Resize(self.image_size),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(
                    mean=self.mean,
                    std=self.std
                )
            ])
        return v2.Compose([
            v2.Resize(self.image_size),
            v2.RandomChoice(
                [
                    v2.RandomHorizontalFlip(),
                    v2.RandomRotation(degrees=(-15,15)),
                    v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                    v2.Lambda(lambda x: x),
                ]
            ),
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
    def regenerate_pair(self):
        self.pairs = self.Create_Pair()
    def __len__(self):
        return self.N
    def __getitem__(self, idx):
        image_path1, image_path2, label = self.pairs[idx]

        #Hiện tại label đang là 0 cho imposter và 1 cho genuine. 
        # Dùng cái này để convert label sang -1 cho imposter và +1 cho genuine cho CosineEmbeddingLoss
        label = label * 2 - 1

        img1 = Image.open(image_path1).convert("RGB")
        img2 = Image.open(image_path2).convert("RGB")

        transform = self.train_transform() if self.transform_type == "train" else self.test_transform()

        img1, img2 = transform(img1), transform(img2)

        return tv_tensors.Image(img1), tv_tensors.Image(img2), torch.tensor(label, dtype=torch.float32)
    
class DatasetLoader():
    def __init__(self, path, std, mean, img_size, batch_size, number_of_sample, transform = False) -> None:
        self.path = path
        self.std = std
        self.mean = mean
        self.img_size = img_size
        self.batch_size = batch_size
        self.number_of_sample = number_of_sample
        self.transform = transform


    def dataset_loader(self, type):
        if type == "train":
            self.training_dataset = SiameseFingerprintDataset(
                path=self.path,
                image_size=self.img_size,
                mean=self.mean,
                std=self.std,
                N=self.number_of_sample,
                transform_type=type,
                enabled_transform=self.transform
            )
            print("Total train image: {0}".format(len(self.training_dataset)))
            loader = DataLoader(
                self.training_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=2,          # START HERE
                pin_memory=True,
                persistent_workers=False, #Chỉnh cái này thành False để tránh hết Ram
                prefetch_factor=2
            )
        else:
            self.testing_dataset = SiameseFingerprintDataset(
                path=self.path,
                image_size=self.img_size,
                mean=self.mean,
                std=self.std,
                N=self.number_of_sample,
                transform_type=type,
                enabled_transform=False
            )
            print("Total test image: {0}".format(len(self.testing_dataset)))
            loader = DataLoader(
                self.testing_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=2,
                pin_memory=True,
                persistent_workers=False #Chỉnh cái này thành False để tránh hết Ram
            )
        print()
        return loader
    def regenerate_pair(self):
        self.training_dataset.regenerate_pair()