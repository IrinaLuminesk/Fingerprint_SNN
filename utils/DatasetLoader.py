from torchvision.transforms import v2
from torchvision import datasets
from torch.utils.data import DataLoader, DistributedSampler
import torch

from utils.Utilities import get_num_workers

class DatasetLoader():
    def __init__(self, path, std, mean, img_size, batch_size, transform = True, distributed = False) -> None:
        self.path = path
        self.std = std
        self.mean = mean
        self.img_size = img_size
        self.batch_size = batch_size
        self.transform = transform
        self.distributed = distributed
    def train_transform(self):
        if self.transform:
            v2.Compose([
                v2.Resize(self.img_size),
                v2.RandomChoice([
                    v2.RandomResizedCrop(size=self.img_size),
                    v2.RandomHorizontalFlip(p=1),
                    v2.RandomVerticalFlip(p=1),
                    v2.Compose([
                        v2.Pad((10, 20)),
                        v2.Resize(self.img_size)
                    ]),
                    v2.Compose([
                        v2.RandomZoomOut(p=1, side_range=(1, 1.5)),
                        v2.Resize(self.img_size)
                    ]),
                    v2.RandomRotation(degrees=(-180, 180)),
                    v2.RandomAffine(degrees=(-180, 180), translate=(0.1, 0.3), scale=(0.5, 1.75)),
                    v2.RandomPerspective(p=1),
                    v2.ElasticTransform(alpha=120),
                    v2.ColorJitter(brightness=(1,2), contrast=(1,2)),
                    v2.RandomPhotometricDistort(brightness=(1,2), contrast=(1,2), p=1),
                    v2.RandomChannelPermutation(),
                    v2.RandomGrayscale(p=1),
                    v2.GaussianBlur(kernel_size=(3, 5), sigma=(0.1, 4.75)),
                    v2.RandomInvert(p=1),
                    v2.Lambda(lambda x: x),
                    ]),
                    v2.ToImage(), 
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize(
                        mean=self.mean,
                        std=self.std
                    )
                ])
        return v2.Compose([
            v2.ToImage(), 
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=self.mean,
                std=self.std
            )
        ])
    def test_transform(self):
        return v2.Compose([
                v2.Resize(self.img_size),
                v2.ToImage(), 
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(
                    mean=self.mean,
                    std=self.std
                )
            ])
    def dataset_loader(self, type):
        if self.distributed == False:
            return self.non_DDP_Loader(type)
        else:
            return self.DDP_Loader(type)
    def non_DDP_Loader(self, type):
        if type == "train":
            training_dataset = datasets.ImageFolder(
                root=self.path,
                transform=self.train_transform()
            )
            print("Total train image: {0}".format(len(training_dataset)))
            loader = DataLoader(
                training_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=2,          # START HERE
                pin_memory=True,
                persistent_workers=False, #Chỉnh cái này thành False để tránh hết Ram
                prefetch_factor=2
            )
        else:
            testing_dataset = datasets.ImageFolder(
                root=self.path,
                transform=self.test_transform()
            )
            print("Total test image: {0}".format(len(testing_dataset)))
            loader = DataLoader(
                testing_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=2,
                pin_memory=True,
                persistent_workers=False #Chỉnh cái này thành False để tránh hết Ram
            )
        print()
        return loader
    def DDP_Loader(self, type):
        WORLD_SIZE = torch.distributed.get_world_size()
        LOCAL_RANK = torch.distributed.get_rank()
        NUM_WORKERS = get_num_workers()
        if type == "train":
            training_dataset = datasets.ImageFolder(
                root=self.path,
                transform=self.train_transform()
            )
            print("Total train image: {0}".format(len(training_dataset)))
            training_sampler = DistributedSampler(
                training_dataset,
                num_replicas=WORLD_SIZE,
                rank=LOCAL_RANK,
                shuffle=True
            )
            loader = DataLoader(
                training_dataset,
                batch_size=self.batch_size,
                sampler=training_sampler,
                num_workers=NUM_WORKERS,
                pin_memory=True,
                drop_last=True,
                shuffle=True
            )
        else:
            testing_dataset = datasets.ImageFolder(
                root=self.path,
                transform=self.test_transform()
            )
            print("Total test image: {0}".format(len(testing_dataset)))
            testing_sampler = DistributedSampler(
                testing_dataset,
                num_replicas=WORLD_SIZE,
                rank=LOCAL_RANK,
                shuffle=False
            )
            loader = DataLoader(
                testing_dataset,
                batch_size=self.batch_size,
                sampler=testing_sampler,
                num_workers=NUM_WORKERS,
                pin_memory=True,
                drop_last=True,
                shuffle=False
            )
        return loader
    def data_transform(self, type):
        if type == "train":
            return self.train_transform()
        else:
            return self.test_transform()
        