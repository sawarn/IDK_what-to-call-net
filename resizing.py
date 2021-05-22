#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 20:48:00 2020

@author: shivam
"""
import glob
import PIL
import os
import matplotlib as plt
import os.path
from PIL import Image

path =r"/home/shivam/Desktop/3bf97246-1516-49bd-b629-0a3ec8b87725.jpeg"
#for i in os.listdir(path):
 #   images_path=glob.glob(os.path.join(path,"*.jpg"))
#for item in images_path:
    #print(images_path)
im = Image.open(path)
f,e = os.path.splitext(path)
imResize = im.resize((250,100),Image.ANTIALIAS)
imResize.save(f,'JPEG', quality = 90)
    