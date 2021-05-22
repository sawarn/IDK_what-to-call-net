# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 11:04:52 2020

@author: Rashi
"""
import os,sys
import glob
import pandas as pd
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import csv
from PIL import Image

mean = 0.5
std = 0.5

class CancerData(Dataset):
    def __init__(self,folder_path,transform = 'None'):
    # initialize the arrays to store the ground truth labels and paths to the images
        self.data = []
        self.breast_density = []
        self.abnormality_type = []
        self.calcification_type = []
        self.calcification_distribution = []
        self.pathology = []
        
    # read the annotations from the CSV file
        with open(folder_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.data.append(row['Images_path'])
                self.breast_density.append([row['breast density']])
                self.abnormality_type.append([row['abnormality type']])
                self.calcification_type.append([row['calc type']])
                self.calcification_distribution.append([row['calc distribution']])
                self.pathology.append([row['pathology']])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
    # take the data sample by its index
        img_path = self.data[idx]
 
    # read image
        img = Image.open(img_path)
 
    # apply the image augmentations if needed
        if self.transform:
            img = self.transform(img)
 
    # return the image and all the associated labels
        dict_data = {
            'img': img,
            'labels': {
                'breast_density': self.breast_density[idx],
                'abnormality_type': self.abnormality_type[idx],
                'calcification_type': self.calcification_type[idx],
                'calcification_distribution': self.calcification_distribution[idx],
                'pathology': self.pathology[idx]
                }
            }
        return dict_data

# if __name__ == "__main__":
#     CancerData(r'D:\Cancer Diagnosis project\calc_case_description_train_set - Copy.csv')
dataset = CancerData(folder_path = r"/home/shivam/Desktop/Python Scripts/Calci_train_csv.csv")
print(dataset)
"""
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.8, 1.2),
                            shear=None, resample=False),
    transforms.RandomRotation(degrees=30),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
"""
