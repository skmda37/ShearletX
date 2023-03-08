import os
import random
import re

import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from PIL import Image

with open('/home/math/kolek/dev/cartoonx++/cartoonx_imagenet/map_clsloc.txt') as f:
    # line.rstrip() is of the form == 'nf0000535 3 turkey'
    lines = [line.rstrip().split(' ') for line in f]
    label_file_dict = dict()
    for l in lines:
        label_file_dict[l[0]] = l[2] # key is l[0] of the form nf0305830 and l[1] is label


    

class Imagenet(Dataset):
    def __init__(self, 
                 imagenet_path='/home/groups/ai/datasets/imagenet_dataset/val/',
                 samples_per_class=10,
                 num_labels=1000):

        
        assert num_labels <= 1000, 'There are 1000 classes in imagenet'
        
        # Get folder names for each label of form n01440764
        label_folders = random.sample(os.listdir(imagenet_path)[:-1], num_labels)
        
        # Full path to image
        fullimg_paths = []
        # Name of label folder
        label_folder_names = []
        # Names of image files
        img_file_names = []
        # Names of labels
        label_names = []
        
        # Add image paths and associated label/file names
        for label in label_folders:
            if not re.match(r'^n\d+', label): # each subfolder should have form "n01944390"
                continue
            path = os.path.join(imagenet_path, label)
            files = os.listdir(path)
            img_files = random.sample(files, min(samples_per_class, len(files)))
            
            for img in img_files:
                fullimg_paths.append(os.path.join(path, img))
                label_folder_names.append(label)
                img_file_names.append(img)
                label_names.append(label_file_dict[label])
        
        self.fullimg_paths = fullimg_paths
        self.label_folder_names = label_folder_names
        self.img_file_names = img_file_names
        self.label_names = label_names
        
        self.convert_to_tensor = transforms.Compose([transforms.ToTensor(),
                                                     transforms.Resize(size=(256,256))])
    
            
    def __len__(self):
        return len(self.fullimg_paths)
    
    
    def __getitem__(self, idx):
        im = Image.open(self.fullimg_paths[idx]).convert('RGB')
        meta_data = {'fullimg_path': self.fullimg_paths[idx], 
                     'label_folder_name': self.label_folder_names[idx],
                     'img_file_name': self.img_file_names[idx], 
                     'label_name': self.label_names[idx]}
        x = self.convert_to_tensor(im).unsqueeze(0)
        return x, meta_data
    
    
    
          
        
        
