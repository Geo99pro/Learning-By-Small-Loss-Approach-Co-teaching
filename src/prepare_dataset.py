import os
import torch
import rasterio
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import Dataset

class PrepareDataset(Dataset):
    def __init__(self,
                train_clean_csv_path: str,
                train_csv_noise_path: str,
                test_csv_path: str,
                augment: bool = False,
                img_size: tuple = (224, 224),
                mean: tuple = (0.485, 0.456, 0.406),
                std: tuple = (0.229, 0.224, 0.225),
                batch_size: int = 32,
                is_change_path: bool = False,
                image_dir: str = None,
                output_dir: str = "output",
                save_cls_distribution: bool = False,
                fix_paths_in_memory: bool = False):

        self.mode = None
        self.augment = augment
        self.img_size = img_size
        self.mean, self.std = mean, std
        self.batch_size = batch_size
        self.is_change_path = is_change_path
        self.image_dir = image_dir
        self.output_dir = output_dir
        self.save_cls_distribution = save_cls_distribution
        self.fix_paths_in_memory = fix_paths_in_memory

        self.df_train_clean = pd.read_csv(train_clean_csv_path)
        self.df_train_noise = pd.read_csv(train_csv_noise_path)
        self.df_test = pd.read_csv(test_csv_path)

    
        if self.is_change_path and self.image_dir: 
            if self.fix_paths_in_memory:
                self.train_clean_csv_path = self._path_fixed(self.df_train_clean)
                self.train_csv_noise_path = self._path_fixed(self.df_train_noise)
                self.test_csv_path = self._path_fixed(self.df_test)
            else:
                self._update_image_paths_on_disk(train_clean_csv_path)
                self._update_image_paths_on_disk(train_csv_noise_path)
                self._update_image_paths_on_disk(test_csv_path)

                self.df_train_clean = pd.read_csv(train_clean_csv_path)
                self.df_train_noise = pd.read_csv(train_csv_noise_path)
                self.df_test = pd.read_csv(test_csv_path)
        self.class_cols_train = self.df_train_clean.columns[2:].tolist()
        assert self.class_cols_train == self.df_train_noise.columns[2:].tolist(), "Class columns in clean and noisy training CSVs do not match."
        self.class_cols_test = self.df_test.columns[2:].tolist()
        self.transforms = None

    def _path_fixed(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if 'img_name' in df.columns:
            df['img_path'] = df['img_name'].apply(lambda x: os.path.join(self.image_dir, os.path.basename(str(x))).replace("\\", "/"))

        elif 'img_path' in df.columns:
            df['img_path'] = df['img_path'].apply(lambda x: os.path.join(self.image_dir, os.path.basename(str(x))).replace("\\", "/"))

        return df

    def _update_image_paths_on_disk(self, csv_file: str):
        """ Update image paths on disk to match the new directory structure."""
        df = pd.read_csv(csv_file)
        df = self._path_fixed(df)
        df.to_csv(csv_file, index=False)

    def _plot_class_distribution(self, df: pd.DataFrame, title: str):
        """ Plot the distribution of classes in the training dataset."""
        fig, ax = plt.subplots(figsize=(12, 6))
        class_counts = df.iloc[:, 2:].sum()
        class_counts.plot(kind='bar', colormap='viridis', ax=ax)
        ax.set_title(title)
        ax.set_xlabel('Classes')
        ax.set_ylabel('Number of Samples')
        fig.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"{title.replace(' ', '_')}.png"), dpi=200)
        plt.close(fig)

    def _build_transforms(self, phase: str = None):
        if self.augment and phase == 'train':
            return transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.RandomRotation(20),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ])
        else:
            return transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ])
    
    def _raster_to_pil_rgb(self, path: str) -> Image.Image:
        with rasterio.open(path) as src:
           img_array = src.read()

           if img_array.shape[0] >= 3:
               img_array = img_array[:3, :, :]

           elif img_array.shape[0] == 1:
               img_array = np.repeat(img_array, 3, axis=0)

           img_array = np.transpose(img_array, (1, 2, 0)).astype('uint8')

           return Image.fromarray(img_array, 'RGB')

    def trainer(self):
        print(f'➡️ In training mode.')
        self.mode = 'train'
        self.nb_class = len(self.class_cols_train)
        print(f'➡️ Number classes (train): {self.nb_class}')
        if self.save_cls_distribution:
            self._plot_class_distribution(self.df_train_clean, title='Class Distribution in Clean Training Data')
            self._plot_class_distribution(self.df_train_noise, title='Class Distribution in Noisy Training Data')
        self.transforms = self._build_transforms(phase='train')
        return self

    def tester(self):
        print(f'➡️ In testing mode.')
        self.mode = 'test'
        self.nb_class = len(self.class_cols_test)
        print(f'➡️ Number classes (test): {self.nb_class}')
        if self.save_cls_distribution:
            self._plot_class_distribution(self.df_test, title='Class Distribution in Test Data')
        self.transforms = self._build_transforms(phase='test')
        return self


    def __len__(self):
        if self.mode is None:
            raise RuntimeError("Dataset mode is not set. Please call 'trainer()' or 'tester()' before using the dataset.")
        if self.mode == 'train':
            return len(self.df_train_clean)
        elif self.mode == 'test':
            return len(self.df_test)
        else:
            raise ValueError("Invalid mode. Mode should be either 'train' or 'test'.")


    def __getitem__(self, idx):
        if self.mode is None:
            raise RuntimeError("Dataset mode is not set. Please call 'trainer()' or 'tester()' before using the dataset.")
        
        if self.mode == 'train':
            row_clean = self.df_train_clean.iloc[idx]
            clean_img = self._raster_to_pil_rgb(row_clean['img_path'])
            clean_img = self.transforms(clean_img)
            label_clean = torch.tensor(row_clean[self.class_cols_train].astype(np.float32).values, dtype=torch.float32)
            row_with_noise = self.df_train_noise.iloc[idx]
            noise_img = self._raster_to_pil_rgb(row_with_noise['img_path'])
            noise_img = self.transforms(noise_img)
            label_with_noise = torch.tensor(row_with_noise[self.class_cols_train].astype(np.float32).values, dtype=torch.float32)
            return clean_img, label_clean, noise_img, label_with_noise
        
        elif self.mode == 'test':
            row = self.df_test.iloc[idx]
            img = self._raster_to_pil_rgb(row['img_path'])
            img = self.transforms(img)
            label = torch.tensor(row[self.class_cols_test].astype(np.float32).values, dtype=torch.float32)
            return img, label

    