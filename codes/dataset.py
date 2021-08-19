# encoding: utf-8

# duplicate, only for debug
"""
Read images and corresponding labels.
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image
import os
import itertools
from torch.utils.data.sampler import Sampler

N_CLASSES = 14
CLASS_NAMES = [
        'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
        'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia', 'No Finding'
                  ]

class NIH(Dataset):
    def __init__(self, dataframe, path_image, transform=None):
        """
        Args:
            data_dir: path to image directory.
            csv_file: path to the file containing images
                with corresponding labels.
            transform: optional transform to be applied on a sample.
        """
        super(NIH, self).__init__()
        self.dataframe = dataframe
        self.path_image = path_image
        self.transform = transform
        

    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            image and its labels
        """
        item = self.dataframe.iloc[index]

        image_name = os.path.join(self.path_image, item["Image Index"])
        image = Image.open(image_name).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        df_label = item['Finding Labels']
        
        if(isinstance(df_label, str)):
            df_label = df_label.split('|') 
            df_label = [CLASS_NAMES.index(i) for i in df_label]

            label = torch.FloatTensor(np.zeros(N_CLASSES, dtype=float))

            for i in df_label:
                if i!=14:
                    label[i] = 1.0
        else:
            label = df_label
        
        return image, label, item["Image Index"]

    def __len__(self):
        return self.dataframe.shape[0]


class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2
        
class TransformOnce:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out = self.transform(inp) 
        return out