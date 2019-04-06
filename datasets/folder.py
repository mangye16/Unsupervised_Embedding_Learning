import torchvision.datasets as datasets

class ImageFolderInstance(datasets.ImageFolder):
    """: Folder datasets which returns the index of the image as well::
    """
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = self.loader(path)
        if len(self.transform.transforms)>4:
            img1 = self.transform(img)
            img2 = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)
            return img1, img2, target, index
        else:
            img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)
            return img, target, index

