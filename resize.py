#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 17:36:33 2020

@author: rashi
"""

import glob
from PIL import Image
import os, sys

images_path = []
path = r"/home/rashi/Documents/Calc"
# for i in os.listdir(path):
#     images_path=glob.glob(os.path.join(path,"*","*","*"))
#     images_path.append(images_path)
#print(images_path)
#for item in os.path.join    
    # for item in images_path:
    #     img = glob.glob(os.path.join(item,"*.png"))
    #     print(img)
    #     # os.path.join(images_path, item)
    #     # im = Image.open(item)
    #     # f, e = os.path.splitext(images_path,item)
    #     # imResize = im.resize((500,500), Image.ANTIALIAS)
    #     # imResize.save(f + ' resized.png', 'PNG', quality=90)
for i in os.listdir(path):
    images_path=glob.glob(os.path.join(path,"*","*","*","*.png"))
    for item in images_path:
        im = Image.open(item)
        f,e = os.path.splitext(item)
        imResize = im.resize((500,500), Image.ANTIALIAS)
        imResize.save(f + 'resized.png', 'PNG', quality = 90)
    