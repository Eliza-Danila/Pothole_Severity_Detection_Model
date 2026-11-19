#This should be implmented with base_cnn 
#It should read the pothole images now instead of generating images since this was a placeholder
#It was supposed to make sure the program ran so now here it is.

import lightning.pytorch as pl
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

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
        
        label_map = {"none": 0, "low": 1, "medium": 2, "high": 3}
        label = label_map[image_path.parent.name.lower()]

        return (image, label)

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

#This is for debugging purposes to make sure everything works
if __name__ == "__main__":

    data_module = PotholeDataModule()

    data_module.setup("fit")
    data_module.setup("test")

    for name, loader in [
        ("Training", data_module.train_dataloader()),
        ("Validation", data_module.val_dataloader()),
        ("Testing", data_module.test_dataloader())
    ]:
        batch = next(iter(loader))
        print(f"{name}: {len(loader.dataset)} images")
        print("Batch shape:", batch[0].shape)

    print("Real image loading successful!")


