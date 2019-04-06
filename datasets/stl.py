from __future__ import print_function
from PIL import Image
import torchvision.datasets as datasets
import torch.utils.data as data
import numpy as np

class STL10Instance(datasets.STL10):
    """CIFAR10Instance Dataset.
    """
    def __getitem__(self, index):
        
        if self.labels is not None:
            img, target = self.data[index], int(self.labels[index])
        else:
            img, target = self.data[index], None

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img1 = self.transform(img)
            if not self.split=='test':
                img2 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if not self.split=='test':
            return img1, img2, target, index
        else:
            return img1, target, index
