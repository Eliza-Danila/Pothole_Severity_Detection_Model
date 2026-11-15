#This basically is a module that takes the images, prepares them and sends them to the neural network
#currently it is using placeholder images, and once I get a dataset, I will update the code accordingly.
#I will also have it implemented to the base_cnn.py at a later date.

import torch
import lightning.pytorch as pl

from torch.utils.data import DataLoader, TensorDataset



class PotholeDataModule(pl.LightningDataModule):

    def __init__(self,
                 batch_size=16,
                 num_workers=0,
                 location="./datasets"):

        super().__init__()

        # Settings
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.location = location

        # Temporary image and label dimensions
        self.input_shape = (3, 64, 64)
        self.output_shape = (3,)

        # Dataset placeholders
        self.data_train = None
        self.data_val = None
        self.data_test = None

        
    def setup(self, stage=None):

        if stage in (None, "fit", "validate"):

            # Create 80 fake training images
            x_train = torch.rand(80, 3, 64, 64)

            # Create 80 random severity labels
            y_train = torch.randint(0, 3, (80,))

            # Create 20 fake validation images
            x_val = torch.rand(20, 3, 64, 64)

            # Create 20 random validation labels
            y_val = torch.randint(0, 3, (20,))

            self.data_train = TensorDataset(
                x_train, y_train
            )

            self.data_val = TensorDataset(
                x_val, y_val
            )

        if stage in (None, "test", "predict"):

            # Create 20 fake testing images
            x_test = torch.rand(20, 3, 64, 64)

            # Create 20 random testing labels
            y_test = torch.randint(0, 3, (20,))

            self.data_test = TensorDataset(
                x_test, y_test
            )


    
    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )

    def predict_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )

