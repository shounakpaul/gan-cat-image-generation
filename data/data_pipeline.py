import os
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


class DataPipeline:
    def __init__(self, dataset_path, image_size, batch_size):
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def get_dataloader(self):
        dataset = ImageFolder(root=self.dataset_path, transform=self.transform)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
