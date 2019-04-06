from __future__ import print_function
from PIL import Image
import torchvision.datasets as datasets
import torch.utils.data as data

class CIFAR10Instance(datasets.CIFAR10):
    """CIFAR10Instance Dataset.
    """
    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        
        if self.target_transform is not None:
            target = self.target_transform(target)
            
        if self.transform is not None:
            img1 = self.transform(img)
            if self.train:
                img2 = self.transform(img)

        if self.train:
            return img1, img2, target, index
        else:
            return img1, target, index