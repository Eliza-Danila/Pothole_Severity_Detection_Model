# THIS PROGRAM IS THE REINFORCED CNN
# This boilerplate was taken from Dr. Joshua L. Phillips course material.

# imports
import numpy as np
import torch
import torchvision
import torchmetrics
import lightning.pytorch as pl
from torchinfo import summary
from torchview import draw_graph
import matplotlib.pyplot as plt
import pandas as pd
import os
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

my_job_id = os.environ["SLURM_JOBID"]

# check if GPU is present
print("GPU is present:",torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU Name:",torch.cuda.get_device_name())
    print("GPU Count:",torch.cuda.device_count())
    print("GPU Properties:",torch.cuda.get_device_properties())

# Function that provides labeling when labels are needed in reward function
def labeling(paths):
    label_map = {"none": 0, "low": 1, "medium": 2, "high": 3}
    labels = []
    for p in paths:
        name = Path(p).parent.name.lower()
        if name not in label_map:
            raise ValueError(f"Unexpected label folder '{name}' for {p}")
        labels.append(label_map[name])
    return torch.tensor(labels)


#Loading Dataset Class
class PotholeImageDataset(Dataset):
    def __init__(self, folder):
        self.folder = Path(folder)
        #Check to see if the folder is there before proceeding
        if not self.folder.is_dir():
            raise FileNotFoundError(f"Dataset folder not found: {self.folder.resolve()}")

        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        self.images = sorted(path for path in self.folder.rglob("*")
            if path.is_file() and path.suffix.lower() in extensions)
        
        if not self.images:
            raise ValueError(f"No images found in {self.folder}")
        #Formats the images to make it readable to the system
        self.transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = self.transform(image)
        return (image, index)

#Prepares the Datasets and DataLoaders for the model
class PotholeDataModule(pl.LightningDataModule):

    def __init__(self, batch_size=16, num_workers=0, location="./datasets"):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.location = Path(location)

        self.input_shape = (3, 64, 64)
        self.output_shape = (4,)

        self.data_train = None
        self.data_val = None
        self.data_test = None

    def setup(self, stage=None):
        #Sets up the testing data
        if stage in (None, "fit", "validate"):
            if self.data_train is None:
                self.data_train = PotholeImageDataset(self.location / "train")

            if self.data_val is None:
                val_folder = self.location / "val"

                if not val_folder.is_dir():
                    val_folder = self.location / "valid"

                self.data_val = PotholeImageDataset(val_folder)
            #Load it when done or predicting
        if stage in (None, "test", "predict"):
            if self.data_test is None:
                self.data_test = PotholeImageDataset(self.location / "test")

    # These are the labels that will only be used in the reward function
    def get_train_labels(self):
        return labeling(self.data_train.images)

    def get_val_labels(self):
        return labeling(self.data_val.images)

    def get_test_labels(self):
        return labeling(self.data_test.images)

    # Sets up the dataloader for train, val, test, and predict
    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True)

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False)

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False)

    def predict_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False)


data_module = PotholeDataModule()
data_module.setup('validate')
data_module.input_shape

input_shape = (1,)+data_module.input_shape
output_shape = (1,)+data_module.output_shape
input_ones = torch.ones(size=input_shape)