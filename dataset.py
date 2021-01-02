"""
    Dataset Classes for loading the GL images from
    fits files

    Based from pytorch "WRITING CUSTOM DATASETS, DATALOADERS AND TRANSFORMS"
    tutorial:
        https://pytorch.org/tutorials/beginner/data_loading_tutorial.html#writing-custom-datasets-dataloaders-and-transforms
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from astropy.utils.data import get_pkg_data_filename
from astropy.io import fits

class SpaceBasedDataset(Dataset):
    """
        Class for loading the Space Based images from the
        Bologna lens finding challenge
    """
    def __init__(self, csv_path, image_folder_path, transform=None):
        """
            Load csv with images's annotations and save
            paths
        """
        df = pd.read_csv(csv_path) 
        # only consider lens images
        self.df = df[df['is_lens'] == 1]
        self.image_folder_path = image_folder_path
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self,idx):
        """
            Return lens "idx" lens image
        """  
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        image_path = '{}/imageEUC_VIS-{}.fits'.format(
            self.image_folder_path,
            self.df.iloc[idx,0] #Get image ID
        )
        # open fits file with astropy
        image = fits.getdata(
            get_pkg_data_filename(image_path),
            ext=0
        )
        
        # Sample for return
        sample = {
            'image': np.reshape(
                image,
                (101,101,1) 
            ) # Consider Channel 
        }
        if self.transform:
            sample = self.transform(sample)
        return sample 

