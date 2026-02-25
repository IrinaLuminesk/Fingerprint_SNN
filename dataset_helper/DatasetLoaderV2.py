from turtle import st
from torchvision.transforms import v2
from torchvision import tv_tensors
from torch.utils.data import DataLoader, Dataset
import torch

from pathlib import Path
from collections import defaultdict
import random
from PIL import Image
from itertools import combinations, product

class SiameseFingerprintDataset(Dataset):
    def __init__(self, path, N, image_size, mean, std, transform_type="train", enabled_transform=False):
        self.path = path #Đường dẫn dataset
        self.N = N #Số lượng pair sẽ được tạo
        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.transform_type = transform_type
        self.enabled_transform = enabled_transform
        
        self.person_fingerprint_ids = self.Create_Person_Fingerprint_Ids()
        self.genuine_pairs = self.Create_Genuine_Pairs()
        self.imposter_pairs = self.Create_Imposter_Pairs()
        print("Total possible pair: {0}".format(len(self.genuine_pairs) + len(self.imposter_pairs)))
        self.pairs = self.Create_Train_Pairs() if self.transform_type == "train" else self.Create_Test_Pairs()
    def Create_Person_Fingerprint_Ids(self):
        person_fingerprint_ids = defaultdict(list)
        for img in Path(self.path).rglob("*"):
            if img.is_file():
                person_fingerprint_ids[img.name.split("_")[0]].append(str(img))
        return person_fingerprint_ids
        #{
        # 1: ["Đường dẫn 1, Đường dẫn 2, ..."]
        # 2: ["Đường dẫn 1, Đường dẫn 2, ..."]
        #}

    def Create_Genuine_Pairs(self):
        genuine_pairs = []
        for _, samples in self.person_fingerprint_ids.items():
            for a, b in combinations(samples, 2):
                genuine_pairs.append((a, b, 1))  # label 1 = genuine
        return genuine_pairs

    def Create_Imposter_Pairs(self):
        imposter_pairs = []
        for (_, s1), (_, s2) in combinations(self.person_fingerprint_ids.items(), 2):
            for a, b in product(s1, s2):
                imposter_pairs.append((a, b, 0))  # label 0 = imposter
        return imposter_pairs

    def Create_Train_Pairs(self):
        half = self.N // 2
        
        genuine_pairs = self.sample_with_limited_duplicates(len(self.genuine_pairs), half)

        imposter_pairs = self.sample_with_limited_duplicates(len(self.imposter_pairs), half)

        pairs = genuine_pairs + imposter_pairs
        
        random.shuffle(pairs)
        return pairs
    def Create_Test_Pairs(self):
        pairs = self.genuine_pairs + self.imposter_pairs
        
        return pairs

    def sample_with_limited_duplicates(self, population, k):
        n = len(population)
        if k <= n:
            return random.sample(population, k)
        
        # Take all elements once
        result = population.copy()
        
        # Add exactly k - n duplicates
        extra = random.choices(population, k=k - n)
        result.extend(extra)
        
        return result
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
        self.pairs = self.Create_Train_Pairs() if self.transform_type == "train" else self.Create_Test_Pairs()
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