import os
import cv2
import torch
import random
import numpy as np
from torchvision import transforms as T
# import matplotlib.pyplot as plt
# from torchvision import transforms

# define a data class
class CamvidDataset(torch.utils.data.Dataset):
    def __init__(self, data, label_dict, transform=None):
        """Define the dataset for classification problems
        Args:
            data ([dataframe]): [a dataframe that contain 2 columns: image name and label]
            transform : [augmentation methods and transformation of images]
            training (bool, optional): []. Defaults to True.
        """
        self.CLASSES = ["sky", "building", "pole", "road", "pavement", \
                        "tree", "signsymbol", "fence", "car",\
                        "pedestrian", "bicyclist", "unlabelled"]
        self.data = data
        self.imgs = data["file_name"].unique().tolist()
        self.transform = transform
        # convert str names to class values on masks, first class is 'background'
        # self.CLASSES = label_dict
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in label_dict]


    def __getitem__(self, idx):
        image = cv2.imread(self.data.iloc[idx, 0])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # extract certain classes from mask
        mask = cv2.imread(self.data.iloc[idx, 1],cv2.IMREAD_UNCHANGED)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')
        
        # t = T.Compose([T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        # apply augmentations
        if self.transform:
            transformed = self.transform(image = image, mask = mask)
            image = transformed['image']
            mask = transformed['mask']

        return image, mask

    def __len__(self):
        return len(self.imgs)
