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
def labeling(labels_csv):
    label_map = {"none": 0, "low": 1, "medium": 2, "high": 3}
    
    df = pd.read_csv(labels_csv)
    
    labels = {}
    for i, row in df.iterrows():
        if row["label"] == "skip":
            continue

        labels[row["filename"]] = label_map[row["label"]]

    return labels


#Loading Dataset Class
class PotholeImageDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.folder = folder
        self.transform = transform

    def __len__(self):
        return len(self.folder)

    def __getitem__(self, index):
        with Image.open(self.folder[index]) as image:
            image = image.convert("RGB")

            if self.transform:
                image = self.transform(image)
        return image, index

# Define image transformations
train_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor()
])

eval_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

#Prepares the Datasets and DataLoaders for the model
class PotholeDataModule(pl.LightningDataModule):

    def __init__(self, batch_size=16, num_workers=0, 
                                    location="./datasets",
                                    reward_csv="./datasets/labels.csv",
                                    split_dir="./datasets/splits"):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.location = Path(location)
        self.reward_csv = Path(reward_csv)
        self.split_dir = Path(split_dir)

        self.input_shape = (3, 64, 64)
        self.output_shape = (4,)

        self.data_train = None
        self.data_val = None
        self.data_test = None
        
        self.data_train_eval = None
        self.labels = {}

        self.splits_created = False

    def setup(self, stage=None):
        if self.splits_created:
            return

        # Map severity labels to numbers
        label_map = {"none": 0, "low": 1, "medium": 2, "high": 3}

        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        # Find all image files
        image_paths = [
            p for p in self.location.rglob("*")
            if p.is_file() and p.suffix.lower() in extensions
        ]

        image_lookup = {p.name: p for p in image_paths}

        datasets = {}

        # Names of every image that is not in the unlabeled pool
        held_out = set()

        # Take from the split csvs
        for split_name in ["val", "test"]:

            df = pd.read_csv(self.split_dir / f"{split_name}.csv")
            
            # Match CSV filenames to images
            paths = []
            labels = []

            for i, row in df.iterrows():
                filename = row["filename"]
                label = str(row["label"]).strip().lower()

                if label not in label_map:
                    raise ValueError(f"Unknown label: {label}")

                if filename not in image_lookup:
                    raise FileNotFoundError(
                        f"Could not find image: {filename}"
                    )

                paths.append(image_lookup[filename])
                labels.append(label_map[label])
                held_out.add(filename)

            datasets[split_name] = PotholeImageDataset(paths, eval_transform)
            self.labels[split_name] = torch.tensor(labels)

        # The first 100 images are the ones used in the reward function
        reward_labels = labeling(self.reward_csv)

        for filename in reward_labels:
            if filename not in image_lookup:
                raise FileNotFoundError(
                    f"Could not find image: {filename}"
                )

        held_out.update(reward_labels)

        # Everything else is the unlabeled pool
        pool = sorted(p for name, p in image_lookup.items() if name not in held_out)
        reward_images = [image_lookup[name] for name in reward_labels]

        train_paths = pool + reward_images
        # Labeling for training (-1 means unlabeled)
        self.labels["train"] = torch.tensor([-1] * len(pool) + list(reward_labels.values()))

        self.data_train = PotholeImageDataset(train_paths, train_transform)
        self.data_train_eval = PotholeImageDataset(train_paths, eval_transform)
        self.data_val = datasets["val"]
        self.data_test = datasets["test"]

        self.splits_created = True

    # These are the labels that will only be used in the reward function
    def get_train_labels(self):
        return self.labels["train"]

    def get_val_labels(self):
        return self.labels["val"]

    def get_test_labels(self):
        return self.labels["test"]

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

# Define model
class NeuralNetwork(pl.LightningModule):
    def __init__(self, input_shape, output_shape,
                 **kwargs):
        super().__init__(**kwargs)
        
        self.input_shape = input_shape
        self.output_shape = output_shape
        
        self.conv1 = torch.nn.Conv2d(3, # Input channels
                                     3, # Output channels
                                     kernel_size=(2,2), # Kernel window
                                     stride=(1,1), # Step size
                                     padding=(0,0))
        self.conv2 = torch.nn.Conv2d(3, # Input channels
                                     5, # Output channels
                                     kernel_size=(2,2), # Kernel window
                                     stride=(1,1), # Step size
                                     padding=(0,0))
        self.conv_activation = torch.nn.ReLU()
        
        self.pool1_2 = torch.nn.MaxPool2d((2,2))
        self.flatten = torch.nn.Flatten()
        self.output_layer = torch.nn.Linear(5*31*31,4)
        
    def forward(self, x):
        y = x
        y = self.conv1(y)
        y = self.conv_activation(y)
        y = self.conv2(y)
        y = self.conv_activation(y)
        y = self.pool1_2(y)
        y = self.flatten(y)
        # y = y.flatten(start_dim=1)
        y = self.output_layer(y)
        return y

    def predict(self, x):
        y = x
        y = self(y)
        y = self.output_activation(y)
        return y

model = NeuralNetwork(input_shape=data_module.input_shape,
                      output_shape=data_module.output_shape)

# show model summary
summary(model, input_size=input_shape)

# Logits
model(input_ones.to(model.device)).cpu().detach()

# Define Trainable Module
class LightningNetwork(NeuralNetwork):
    def __init__(self, **kwargs):
        # This is the contructor, where we typically make
        # layer objects using provided arguments.
        super().__init__(**kwargs) # Call the super class constructor
        
    def predict_step(self, predict_batch, batch_idx):
        x, y_true = predict_batch
        y_pred = self.predict(x)
        return y_pred, y_true

    def training_step(self, train_batch, batch_idx):
        x, y_true = train_batch
        y_pred = self(x)
        for metric_name, metric_function in self.network_metrics.items():
            metric_value = metric_function(y_pred,y_true)
            self.log('train_'+metric_name, metric_value, on_step=False, on_epoch=True)
        loss = self.network_loss(y_pred,y_true)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss
        
    def validation_step(self, val_batch, batch_idx):
        x, y_true = val_batch
        y_pred = self(x)
        for metric_name, metric_function in self.network_metrics.items():
            metric_value = metric_function(y_pred,y_true)
            self.log('val_'+metric_name, metric_value, on_step=False, on_epoch=True)
        loss = self.network_loss(y_pred,y_true)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        return loss
        
    def test_step(self, test_batch, batch_idx):
        x, y_true = test_batch
        y_pred = self(x)
        for metric_name, metric_function in self.network_metrics.items():
            metric_value = metric_function(y_pred,y_true)
            self.log('test_'+metric_name, metric_value, on_step=False, on_epoch=True)
        loss = self.network_loss(y_pred,y_true)
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        return loss

# Define Trainable Module - COME BACK TO
class ReinforcedNetwork(LightningNetwork):
    def __init__(self, **kwargs):
        # This is the contructor, where we typically make
        # layer objects using provided arguments.
        super().__init__(**kwargs) # Call the super class constructor

        # This creates an accuracy function
        self.network_metrics = torch.nn.ModuleDict({
            'acc': torchmetrics.classification.Accuracy(task='multiclass',
                                                        num_classes=self.output_shape[-1])
        })
        # This creates a loss function
        self.network_loss = torch.nn.CrossEntropyLoss()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

model = ReinforcedNetwork(input_shape=data_module.input_shape,
                          output_shape=data_module.output_shape)

# summary of model
summary(model, input_size=input_shape)

# log the data
logger = pl.loggers.CSVLogger("logs",
                              name="Reinforced-CNN",
                              version=my_job_id)

wandb_logger = pl.loggers.WandbLogger(project="Reinforced-CNN",name= my_job_id)

trainer = pl.Trainer(logger=[logger,wandb_logger],
                     max_epochs=2,
                     enable_progress_bar=True,
                     enable_checkpointing=False,
                     log_every_n_steps=0,
                     callbacks=[pl.callbacks.RichProgressBar(refresh_rate=20)])

trainer.fit(model, data_module)

trainer.validate(model, data_module)

trainer.test(model, data_module)

results = pd.read_csv(logger.log_dir+"/metrics.csv")
results

print("Validation accuracy:",*["%.8f"%(x) for x in
                               results['val_acc'][np.logical_not(np.isnan(results["val_acc"]))]])
print("Validation loss:",*["%.8f"%(x) for x in
                               results['val_loss'][np.logical_not(np.isnan(results["val_loss"]))]])

print("Training accuracy:", *["%.8f"%(x) for x in
    results['train_acc'][np.logical_not(np.isnan(results["train_acc"]))]])

print("Training loss:", *["%.8f"%(x) for x in
    results['train_loss'][np.logical_not(np.isnan(results["train_loss"]))]])