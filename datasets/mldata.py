from __future__ import print_function
from PIL import Image
import torchvision.datasets as datasets
import torch.utils.data as data
import numpy as np
from torchvision import transforms

        
class MLDataInstance(data.Dataset):
    """Metric Learning Dataset.
    """
    def __init__(self, src_dir, dataset_name, train = True, transform=None, target_transform=None, nnIndex = None):
       
        data_dir = src_dir + dataset_name + '/'
        if train:
            img_data  = np.load(data_dir + '{}_{}_256resized_img.npy'.format('training',dataset_name))
            img_label = np.load(data_dir + '{}_{}_256resized_label.npy'.format('training',dataset_name))
        else:
            img_data  = np.load(data_dir + '{}_{}_256resized_img.npy'.format('validation',dataset_name))
            img_label = np.load(data_dir + '{}_{}_256resized_label.npy'.format('validation',dataset_name))

        self.img_data  = img_data
        self.img_label = img_label
        self.transform = transform
        self.target_transform = target_transform
        self.nnIndex = nnIndex

    def __getitem__(self, index):
        
        if self.nnIndex is not None:

            img1, img2, target = self.img_data[index], self.img_data[self.nnIndex[index]], self.img_label[index]

            img1 = self.transform(img1)
            img2 = self.transform(img2)
            if self.target_transform is not None:
                target = self.target_transform(target)
            return img1, img2, target, index
            
        else:
            img, target = self.img_data[index], self.img_label[index]
            img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)

            return img, target, index
        
    def __len__(self):
        return len(self.img_data)