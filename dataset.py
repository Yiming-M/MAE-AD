import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms
import os
from PIL import Image

from typing import Union, Tuple


_CLASS_NAMES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]


class MVTecDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        category: str,
        resize: Union[int, Tuple[int, int]] = 256,
        crop_size: Union[int, Tuple[int, int]] = 224,
    )-> None:
        super().__init__()
        assert split in ["train", "test"] and category in _CLASS_NAMES
        self.root = os.path.join(root, category, split)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(resize, antialias=True),
            transforms.CenterCrop(crop_size),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self._make_data()

    def _make_data(self) -> None:
        self.img_paths = []
        self.labels = []
        dir_names = os.listdir(self.root)
        for dir_name in dir_names:
            img_names = os.listdir(os.path.join(self.root, dir_name))
            for img_name in img_names:
                self.img_paths.append(os.path.join(self.root, dir_name, img_name))
                self.labels.append(0 if dir_name == "good" else 1)
    
    def __len__(self) -> int:
        return len(self.img_paths)
    
    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        with Image.open(self.img_paths[idx]) as img:
            img = img.convert("RGB")
        img = self.transform(img)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, label
