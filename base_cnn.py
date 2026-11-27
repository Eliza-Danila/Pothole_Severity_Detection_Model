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
from torchvision.models import resnet50, ResNet50_Weights

my_job_id = os.environ["SLURM_JOBID"]

# check if GPU is present
print("GPU is present:",torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU Name:",torch.cuda.get_device_name())
    print("GPU Count:",torch.cuda.device_count())
    print("GPU Properties:",torch.cuda.get_device_properties())


#Loading Dataset Class
class PotholeImageDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.folder = folder
        self.transform = transform

    def __len__(self):
        return len(self.folder)

    def __getitem__(self, index):
        image_path, label = self.folder[index]

        with Image.open(image_path) as image:
            image = image.convert("RGB")

            if self.transform:
                image = self.transform(image)
        return image, label

IMG_SIZE = 224  # ResNet's native size

imagenet_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(0.3, 0.3, 0.3),
    transforms.ToTensor(),
    imagenet_norm,
])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    imagenet_norm,
])

#Prepares the Datasets and DataLoaders for the model
class PotholeDataModule(pl.LightningDataModule):

    def __init__(self, batch_size=32, num_workers=4, location="./datasets/label_more_sample", split_dir="./datasets/splits"):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.location = Path(location)
        self.split_dir = Path(split_dir)

        self.input_shape = (3, IMG_SIZE, IMG_SIZE)
        self.output_shape = (4,)

        self.data_train = None
        self.data_val = None
        self.data_test = None

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

        # Take from the split csvs
        for split_name, transform in [("train", train_transform),
                                      ("val", eval_transform),
                                      ("test", eval_transform)]:

            df = pd.read_csv(self.split_dir / f"{split_name}.csv")
            
            # Match CSV filenames to images
            records = []

            for i, row in df.iterrows():
                filename = row["filename"]
                label = str(row["label"]).strip().lower()

                if label not in label_map:
                    raise ValueError(f"Unknown label: {label}")

                if filename not in image_lookup:
                    raise FileNotFoundError(
                        f"Could not find image: {filename}"
                    )

                records.append((
                    image_lookup[filename],
                    label_map[label]
                ))

            datasets[split_name] = PotholeImageDataset(records, transform)

        self.data_train = datasets["train"]
        self.data_val = datasets["val"]
        self.data_test = datasets["test"]

        self.splits_created = True

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
    def __init__(self, input_shape, output_shape, freeze_backbone=True, **kwargs):
        super().__init__(**kwargs)

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.freeze_backbone = freeze_backbone

        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        self.backbone.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(self.backbone.fc.in_features, output_shape[-1]))

        self.output_activation = torch.nn.Softmax(dim=1)

        if freeze_backbone:
            for name, p in self.backbone.named_parameters():
                if not name.startswith("fc."):
                    p.requires_grad = False

    def train(self, mode=True):
        # keep frozen BatchNorm layers using their pretrained statistics
        super().train(mode)
        if self.freeze_backbone:
            for name, module in self.backbone.named_children():
                if name != "fc":
                    module.eval()
        return self

    def forward(self, x):
        return self.backbone(x)

    def predict(self, x):
        return self.output_activation(self(x))

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
        
    def test_step(self, test_batch, batch_idx):
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
        self.network_loss = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        return torch.optim.AdamW(params, lr=1e-5, weight_decay=1e-2)

model = MultiClassNetwork(input_shape=data_module.input_shape,
                          output_shape=data_module.output_shape,
                          freeze_backbone=True)

# summary of model
summary(model, input_size=input_shape)

# log the data
logger = pl.loggers.CSVLogger("logs",
                              name="Base-CNN",
                              version=my_job_id)

wandb_logger = pl.loggers.WandbLogger(project="Base-CNN",name= my_job_id)

trainer = pl.Trainer(logger=[logger,wandb_logger],
                     max_epochs=300,
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