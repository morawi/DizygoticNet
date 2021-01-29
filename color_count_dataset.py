# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 14:10:34 2021

@author: malrawi


"""

# import glob
# import os
from torch.utils.data import Dataset # Dataset class from PyTorch
from PIL import Image, ImageOps, ImageChops # PIL is a nice Python Image Library that we can use to handle images

from file_io import read_from_json
       

class ColorCountDataset(Dataset):
    def __init__(self, root='../data/ColorCountData/', 
                 mode="train",  HPC_run=False, 
                 transforms=None):        
               
        if HPC_run:
            root = '/home/malrawi/MyPrograms/Data/ColorCountData/'
        
        self.transforms = transforms
        self.root = root+mode+'/'
        self.data_dict = read_from_json(self.root+'colors_of_fashion.json')
        self.files_names = list(self.data_dict['colors'].keys())
        self.classes = list(self.data_dict['all_labels'])
        
    def number_of_classes(self, opt):
        return(len(self.classes)) # this should do
  

    def __getitem__(self, index):   
                
        fname = self.files_names[index % len(self.files_names)]
        image_p = Image.open(self.root+fname) # read the image, according to the file name, index select which image to read; index=1 means get the first image in the list self.files_A
        labels = self.data_dict['labels'][fname]
        colors = self.data_dict['colors'][fname]
        mask = image_p.getchannel('A') # getting the mask from the alpha channel (png image), 1 for foreground, 0 otherwise
        image_p = image_p.convert('RGB')
        image_n = ImageOps.invert(image_p)
        image_n = ImageChops.multiply(image_n, mask.convert('RGB'))
        
        if self.transforms is not None:
            image_p = self.transforms(image_p)
            image_n = self.transforms(image_n)
        num_colors = len(labels)
        return image_p, image_n, colors, labels, num_colors
    
    def __len__(self): # this function returns the length of the dataset, the source might not equal the target if the data is unaligned
        return len(self.files_names)

xx = ColorCountDataset()
