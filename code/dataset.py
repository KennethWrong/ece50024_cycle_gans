import os
import numpy as np
import torch
import torchvision as tv
from torchvision.io import read_image
from PIL import Image, ImageFile
import pandas as pd
from torch.utils.data import Dataset
from torchvision import datasets
import re
import cv2
import glob
import random

ImageFile.LOAD_TRUNCATED_IMAGES = True

class ImageDataset(Dataset):
    def __init__(self, root, mode, paired=False):

        self.monet_images = sorted(glob.glob(os.path.join(root, f"{mode}_monet") + "/*.*"))
        self.img_height = 224
        self.img_width = 224
        self.nature_images = sorted(glob.glob(os.path.join(root, f"{mode}_nature") + "/*.*"))
        self.paired = paired
        self.transform = tv.transforms.Compose([
            tv.transforms.Resize(int(self.img_height * 1.12), cv2.INTER_CUBIC),
            tv.transforms.RandomCrop((self.img_height, self.img_width)),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    
    # 1. we got to load the images from .jpg to PIL image or np.array
    def load_image(self, filename):
        # image = Image.open(filename)
        # image.show()
        # image = image.convert('RGB')
        # transform = tv.transforms.PILToTensor()
        # image = transform(image)

        image = cv2.imread(filename)
        transform = tv.transforms.ToTensor()
        image = transform(image)
        

        return image
    
    
    def __len__(self):
        return max(len(self.monet_images), len(self.nature_images))
    
    def __getitem__(self, idx):
        monet_image = self.load_image(self.monet_images[idx % len(self.monet_images)])
        nature_image = None

        if self.paired:
            nature_image = self.load_image(self.nature_images[idx % len(self.nature_images)])
        else:
            nature_image = self.load_image(self.nature_images[random.randint(0, len(self.nature_images) - 1)])
        
        if self.transform:
            monet_image = self.transform(monet_image)
            nature_image = self.transform(nature_image)
        
        return {"M": monet_image, "N": nature_image}