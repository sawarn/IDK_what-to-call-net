#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 19:50:21 2020

@author: shivam
"""
import torch
import torch.nn
from torch.autograd import Variable
import torch.nn.functional as F

class SimpleCNN(torch.nn.Module):
   def __init__(self,abnormality_class=1,pathology_class=3,calc_class=45,breast_class=4,calc_dist_class=9):
      super(SimpleCNN, self).__init__()
      
      self.Convlayer1 = torch.nn.Sequential(
            torch.nn.Conv2d_1(1, 8, kernel_size=3, stride=1, padding=1),
            torch.nn.PReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Dropout())
      
      self.Convlayer2 = torch.nn.Sequential(
            torch.nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(8),
            torch.nn.PReLU(),
            torch.nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=0),
            torch.nn.PReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            
            torch.nn.Dropout())
      
      
      self.Convlayer3 = torch.nn.Sequential(
              torch.nn.Conv2d(16, 16, 3, stride=1, padding=1),
              torch.nn.BatchNorm2d(16),
              torch.nn.PReLU(),
              torch.nn.Conv2d(16,32,3,stride=1,padding=0),
              torch.nn.PReLU(),
              torch.nn.Conv2d(32,64,3,stride=1,padding=0),
              torch.nn.BatchNorm2d(64),
              torch.nn.PReLU(),
              torch.nn.MaxPool2d(kernel_size=2),
              torch.nn.Dropout()
              )
      
     
      self.Convlayer4= torch.nn.Sequential(
          
          torch.nn.Conv2d(64, 64, 3,stride=1,padding=1),
          torch.nn.PReLU(),
          torch.nn.Conv2d(64, 128, 3,stride=1,padding=0),
          torch.nn.BatchNorm2d(128),
          torch.nn.PReLU(),
          torch.nn.Conv2d(128, 256, 3,stride=1,padding=0),
          torch.nn.PReLU(),
          torch.nn.Conv2d(256, 512, 3,stride=1,padding=0),
          torch.nn.BatchNorm2d(512),
          torch.nn.PReLU(),
          torch.nn.MaxPool2d(3),
          torch.nn.Dropout()
          )
      self.Lin1=torch.nn.Linear(512*18*18,abnormality_class)
      self.Lin2=torch.nn.Linear(512*18*18,calc_class)
      self.Lin3=torch.nn.Linear(512*18*18,calc_dist_class)
      self.Lin4=torch.nn.Linear(512*18*18,pathology_class)
      self.Lin5=torch.nn.Linear(512*18*18,breast_class)
      self.Softmax=torch.nn.Softmax()
          
   def forward(self, x):
       x=self.Convlayer1(x)
       x=self.Convlayer2(x)
       x=self.Convlayer3(x)
       x=self.Convlayer4(x)
       x=x.view(-1,512*18*18)
       return{
       'Abnormality:':self.Lin1(x),
       'Calcification Type:':self.Lin2(x),
       'Calcification Distribution:':self.Lin3(x),
       'Pathology:':self.Lin4(x),
       'Breast Density:':self.Lin5(x)
       }

