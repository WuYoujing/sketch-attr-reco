import json
from torchvision import transforms
import math
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os


class FS2KDATA(Dataset):
    def __init__(self, label_path, transform, mode="train"):
        self.transform = transform
        with open(label_path) as f:
            row_data = json.load(f)
        self.img_path = []
        self.labels = []
        for item in row_data:
            str = item['image_name'].replace('photo', 'sketch').replace('image', 'sketch')
            str += '.jpg'
            self.img_path.append(str)
            self.labels.append(item['hair'])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img_path = self.img_path[index]
        label = self.labels[index]
        img = Image.open(os.path.join('FS2K/sketch', img_path)).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def set_traansform():
    transform = []
    transform.append(transforms.Resize(size=(224, 224)))
    transform.append(transforms.ToTensor())
    transform.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
    transform = transforms.Compose(transform)
    return transform


def get_loader(label_path, batch_size, mode='train', transform=None):
    dataset = FS2KDATA(label_path, transform, mode)
    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=(mode == 'train'),
                             drop_last=True)
    return data_loader
