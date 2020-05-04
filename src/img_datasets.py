import torch.utils.data as data
import torch
from PIL import Image
from glob import glob
import torchvision.transforms as transforms
import numpy as np
import pandas as pd

class CovidNetDataset(data.Dataset):

    def __init__(self, config):

        file_path = config["file_path"]
        self.df = self.get_df(file_path)
        self.root_path = config["root_path"]
        self.img_index = 1
        self.label_index = 2

        self.transforms = config["transforms"]
        self.to_pil = transforms.ToPILImage()
        self.label2vec = {"pneumonia": 1, "normal": 0, "COVID-19": 1}

    def get_df(self, path):
        df = pd.read_csv(path, delimiter=' ', error_bad_lines=False, header=None)
        return df

    def load_img(self, filepath, file_format="png"):
        if file_format in ["png", "jpg", "jpeg"]:
            img = Image.open(self.root_path + filepath)
        else:
            print("Unknown image format")
            exit()
        return img

    def preprocess(self, img):
        "Makes number of channels consistent"
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        elif img.shape[0] == 4:
            img = img[:3, :, :]
        return img

    def __getitem__(self, index):
        # img = self.load_img(self.img_paths[index], file_format=self.img_fmt)
        img_path = self.df.iloc[index][self.img_index]
        img = self.load_img(img_path)
        label = self.label2vec[self.df.iloc[index][self.label_index]]
        if self.transforms is not None:
            img = self.transforms(img)
        img = self.preprocess(img)
        return img, label

    def __len__(self):
        return len(self.df)
