"""PyTorch dataset for NIH ChestX-ray14 using the split CSVs from data_prep.py."""
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

FINDINGS = ["Cardiomegaly", "Effusion", "Atelectasis", "Pneumothorax", "Edema"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_image_index(data_dir: Path) -> dict:
    """Map file name -> full path. NIH images are spread over images_001 ... images_012."""
    paths = {p.name: p for p in Path(data_dir).rglob("*.png")}
    if not paths:
        raise FileNotFoundError(f"No .png images found under {data_dir}")
    return paths


def get_transforms(img_size: int, train: bool):
    if train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomRotation(7),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class ChestXrayDataset(Dataset):
    def __init__(self, csv_path, image_index: dict, img_size=224, train=False):
        self.df = pd.read_csv(csv_path)
        missing = ~self.df["image_index"].isin(image_index.keys())
        if missing.any():
            print(f"Warning: {missing.sum()} images in {csv_path} not found, skipping them")
            self.df = self.df[~missing].reset_index(drop=True)
        self.paths = [image_index[n] for n in self.df["image_index"]]
        self.labels = torch.tensor(self.df[FINDINGS].values, dtype=torch.float32)
        self.tf = get_transforms(img_size, train)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        # NIH images are 8-bit grayscale; repeat to 3 channels for an ImageNet backbone.
        img = Image.open(self.paths[i]).convert("RGB")
        return self.tf(img), self.labels[i]

    def pos_weight(self):
        """negatives / positives per finding, to counter class imbalance in the loss."""
        pos = self.labels.sum(0).clamp(min=1)
        return (len(self.labels) - pos) / pos
