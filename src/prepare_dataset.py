import os
import torch
import rasterio
import numpy as np
import pandas as pd
from enum import Enum
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import Dataset

IMG_PATH_COL = 'img_path'
IMG_NAME_COL = 'img_name'

class StandardParams(Enum):
    IMG_SIZE = (224, 224)
    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)
    BATCH_SIZE = 64

class Utils:
    @staticmethod
    def validate_file(file_path: str) -> bool:
        """ Check if a file exists at the given path."""
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
    
    @staticmethod
    def path_fixed(df: pd.DataFrame, image_dir: str) -> pd.DataFrame:
        df = df.copy()
        if IMG_NAME_COL in df.columns:
            df[IMG_PATH_COL] = df[IMG_NAME_COL].apply(lambda x: os.path.join(image_dir, os.path.basename(str(x))).replace("\\", "/"))
        
        elif IMG_PATH_COL in df.columns:
            df[IMG_PATH_COL] = df[IMG_PATH_COL].apply(lambda x: os.path.join(image_dir, os.path.basename(str(x))).replace("\\", "/"))
        return df
    
    @staticmethod
    def update_image_paths_on_disk(csv_file: str, image_dir: str):
        """ Update image paths on disk to match the new directory structure."""
        Utils.validate_file(csv_file)
        df = pd.read_csv(csv_file)
        df = Utils.path_fixed(df, image_dir)
        df.to_csv(csv_file, index=False)

    @staticmethod
    def plot_class_distribution(df: pd.DataFrame, title: str, output_dir: str):
        """ Plot the distribution of classes in the training dataset.
        
        Args:
            df (pd.DataFrame): DataFrame containing the dataset.
            title (str): Title for the plot.
            output_dir (str): Directory to save the plot.
        
        Returns:
            None. However, saves the plot as a PNG file in the specified output directory.
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        class_counts = df.iloc[:, 2:].sum()
        class_counts.plot(kind='bar', colormap='viridis', ax=ax)
        ax.set_title(title)
        ax.set_xlabel('Classes')
        ax.set_ylabel('Number of Samples')
        fig.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{title.replace(' ', '_')}.png"), dpi=200)
        plt.close(fig)

    @staticmethod
    def build_transforms(augment: bool, 
                    img_size: tuple, 
                    mean: tuple, 
                    std: tuple, 
                    phase: str = None) -> transforms.Compose:
        """
        Build image transformations.
        
        Args:
            augment (bool): Whether to apply data augmentation.
            img_size (tuple): Desired image size (width, height).
            mean (tuple): Mean for normalization.
            std (tuple): Standard deviation for normalization.
            phase (str): 'train' or 'test' to specify the phase.

        Returns:
            torchvision.transforms.Compose: Composed transformations.
        """
        transform_list = [transforms.Resize(img_size), transforms.ToTensor(), transforms.Normalize(mean, std)]
        if augment and phase == 'train':
            transform_list.insert(1, transforms.RandomRotation(20))
            transform_list.insert(2, transforms.RandomHorizontalFlip())
        return transforms.Compose(transform_list)

    @staticmethod
    def raster_to_pil_rgb(path: str) -> Image.Image:
        Utils.validate_file(path)
        with rasterio.open(path) as src:
           img_array = src.read()
           if img_array.shape[0] >= 3:
               img_array = img_array[:3, :, :]
           elif img_array.shape[0] == 1:
               img_array = np.repeat(img_array, 3, axis=0)
           img_array = np.transpose(img_array, (1, 2, 0)).astype('uint8')
           return Image.fromarray(img_array, 'RGB')

class PrepareDatasetBase(Dataset):
    """ Base class for preparing datasets."""
    def __init__(self,
                csv_path: str, 
                is_change_path: bool, 
                image_dir: str, 
                output_dir: str, 
                save_cls_distribution: bool, 
                fix_paths_in_memory: bool):
        
        self.is_change_path = is_change_path
        self.image_dir = image_dir
        self.output_dir = output_dir
        self.save_cls_distribution = save_cls_distribution
        self.fix_paths_in_memory = fix_paths_in_memory

        Utils.validate_file(csv_path)
        self.df = pd.read_csv(csv_path)

        if self.is_change_path and self.image_dir:
            if self.fix_paths_in_memory:
                self.df = Utils.path_fixed(self.df, self.image_dir)
            else:
                Utils.update_image_paths_on_disk(csv_path, self.image_dir)
                self.df = pd.read_csv(csv_path)

        self.class_cols = self.df.columns[2:].tolist()
        print(f'➡️ Number of classes: {len(self.class_cols)}')
        self.transforms = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Utils.raster_to_pil_rgb(row[IMG_PATH_COL])
        img = self.transforms(img)
        label = torch.tensor(row[self.class_cols].astype(np.float32).values, dtype=torch.float32)
        return img, label

class PrepareTrainDataset(PrepareDatasetBase):
    def __init__(self,
                 train_clean_csv_path: str,
                 train_noise_csv_path: str,
                 augment: bool = False,
                 **kwargs):
        super().__init__(train_clean_csv_path, **kwargs)
        Utils.validate_file(train_noise_csv_path)
        Utils.update_image_paths_on_disk(train_noise_csv_path, self.image_dir) if self.is_change_path and self.image_dir and not self.fix_paths_in_memory else None
        self.df_noise = pd.read_csv(train_noise_csv_path)
        assert self.class_cols == self.df_noise.columns[2:].tolist(), "Class columns in clean and noisy CSVs do not match."
        self.augment = augment

    def trainer(self):
        print(f'➡️ In training mode.')
        self.nb_class = len(self.class_cols)
        print(f'➡️ Number classes: {self.nb_class}')
        if self.save_cls_distribution:
            Utils.plot_class_distribution(self.df, title='Class Distribution in Clean Training Data', output_dir=self.output_dir)
            Utils.plot_class_distribution(self.df_noise, title='Class Distribution in Noisy Training Data', output_dir=self.output_dir)
        self.transforms = Utils.build_transforms(self.augment, StandardParams.IMG_SIZE.value, StandardParams.MEAN.value, StandardParams.STD.value, phase='train')
        return self
    
    def __getitem__(self, idx):
        img, label_clean = super().__getitem__(idx)
        row_with_noise = self.df_noise.iloc[idx]
        noise_img = Utils.raster_to_pil_rgb(row_with_noise[IMG_PATH_COL])
        noise_img = self.transforms(noise_img)
        label_noise = torch.tensor(row_with_noise[self.class_cols].astype(np.float32).values, dtype=torch.float32)
        return noise_img, label_noise, label_clean
    
class PrepareTestDataset(PrepareDatasetBase):
    def tester(self):
        print('➡️ In testing mode.')
        if self.save_cls_distribution:
            Utils.plot_class_distribution(self.df, title='Class Distribution in Test Data', output_dir=self.output_dir)
        self.transforms = Utils.build_transforms(False, StandardParams.IMG_SIZE.value, StandardParams.MEAN.value, StandardParams.STD.value, phase='test')
        return self
    

# if __name__ == "__main__":
#     from torch.utils.data import DataLoader
#     train_class = PrepareTrainDataset(train_clean_csv_path="D:/meus_codigos_doutourado/Doutourado_2025_2/clean_code/Learning-By-Small-Loss-Approach-Co-teaching/dataset/arvore_dataset_train_clean.csv",
#                                       train_noise_csv_path="D:/meus_codigos_doutourado/Doutourado_2025_2/clean_code/Learning-By-Small-Loss-Approach-Co-teaching/dataset/arvore_dataset_train_noise_25.csv",
#                                       augment=True,
#                                       is_change_path=True,
#                                       image_dir="D:/meus_codigos_doutourado/Doutourado_2025_2/s1/s1/200m",
#                                       output_dir="D:/meus_codigos_doutourado/Doutourado_2025_2/clean_code/Learning-By-Small-Loss-Approach-Co-teaching/output",
#                                       save_cls_distribution=False,
#                                       fix_paths_in_memory=False)
#     test_class = PrepareTestDataset(csv_path="D:/meus_codigos_doutourado/Doutourado_2025_2/clean_code/Learning-By-Small-Loss-Approach-Co-teaching/dataset/arvore_dataset_test_clean.csv",
#                                    is_change_path=True,
#                                    image_dir="D:/meus_codigos_doutourado/Doutourado_2025_2/s1/s1/200m",
#                                    output_dir="D:/meus_codigos_doutourado/Doutourado_2025_2/clean_code/Learning-By-Small-Loss-Approach-Co-teaching/output",
#                                    save_cls_distribution=False,
#                                    fix_paths_in_memory=False)
    
    
#     train_dataset = train_class.trainer()
#     test_dataset = test_class.tester()

#     train_loader = DataLoader(train_dataset, batch_size=StandardParams.BATCH_SIZE.value, shuffle=True)
#     test_loader = DataLoader(test_dataset, batch_size=StandardParams.BATCH_SIZE.value, shuffle=False)

#     for images, labels_noise, labels_clean in train_loader:
#         print(f'Images batch shape: {images.size()}')
#         print(f'Labels (noise) batch shape: {labels_noise.size()}')
#         print(f'Labels (clean) batch shape: {labels_clean.size()}')
#         break

#     for images, labels in test_loader:
#         print(f'Images batch shape: {images.size()}')
#         print(f'Labels batch shape: {labels.size()}')
#         break
