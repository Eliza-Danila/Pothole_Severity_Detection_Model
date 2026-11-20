# THIS PROGRAM IS A BASE COMPARISON TO THE REINFORCED CNN
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
        
        self.labels = labeling(self.images)

        #Formats the images to make it readable to the system
        self.transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = self.transform(image)
        return (image, self.labels[index])

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

# Define model - SIZING OF THE CONV AND POOLING WILL CHANGE BASED ON DATASET
class NeuralNetwork(pl.LightningModule):
    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)
        self.conv1 = torch.nn.Conv2d(1, # Input channels
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
        self.output_layer = torch.nn.Linear(20,10)
        self.output_activation = torch.nn.Softmax()
        
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

# Probabilities
model.predict(input_ones.to(model.device)).cpu().detach()

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
        
    def testing_step(self, test_batch, batch_idx):
        x, y_true = test_batch
        y_pred = self(x)
        for metric_name, metric_function in self.network_metrics.items():
            metric_value = metric_function(y_pred,y_true)
            self.log('test_'+metric_name, metric_value, on_step=False, on_epoch=True)
        loss = self.network_loss(y_pred,y_true)
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        return loss

# Define Trainable Module
class MultiClassNetwork(LightningNetwork):
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
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        return optimizer

model = MultiClassNetwork(input_shape=data_module.input_shape,
                          output_shape=data_module.output_shape)

# summary of model
summary(model, input_size=input_shape)

# log the data
logger = pl.loggers.CSVLogger("logs",
                              name="Base-CNN",
                              version=my_job_id)

trainer = pl.Trainer(logger=logger,
                     max_epochs=50,
                     enable_progress_bar=True,
                     enable_checkpointing=False,
                     log_every_n_steps=0,
                     callbacks=[pl.callbacks.RichProgressBar(refresh_rate=20)])

trainer.validate(model, data_module)

trainer.fit(model, data_module)

trainer.validate(model, data_module)

results = pd.read_csv(logger.log_dir+"/metrics.csv")
results

print("Validation accuracy:",*["%.8f"%(x) for x in
                               results['val_acc'][np.logical_not(np.isnan(results["val_acc"]))]])
print("Validation loss:",*["%.8f"%(x) for x in
                               results['val_loss'][np.logical_not(np.isnan(results["val_loss"]))]])