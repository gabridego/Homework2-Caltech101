from torchvision.datasets import VisionDataset

from PIL import Image

import os
import os.path
import sys

def make_dataset(directory, class_to_idx, split_file):
    instances = []
    keys = class_to_idx.keys()
    with open(split_file) as f:
        split_paths = [line.rstrip() for line in f]

    for path in split_paths:
        full_path = os.path.join(directory, path)
        target_class = path.split('/')[0]
        if os.path.exists(full_path) and target_class in keys:
            item = pil_loader(full_path), class_to_idx[target_class]
            instances.append(item)

    return instances

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class Caltech(VisionDataset):
    def __init__(self, root, split='train', transform=None, target_transform=None):
        super(Caltech, self).__init__(root, transform=transform, target_transform=target_transform)

        self.split = split # This defines the split you are going to use
                           # (split files are called 'train.txt' and 'test.txt')

        '''
        - Here you should implement the logic for reading the splits files and accessing elements
        - If the RAM size allows it, it is faster to store all data in memory
        - PyTorch Dataset classes use indexes to read elements
        - You should provide a way for the __getitem__ method to access the image-label pair
          through the index
        - Labels should start from 0, so for Caltech you will have lables 0...100 (excluding the background class) 

        Based on ImageFolder dataset (https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py)
        '''
        classes, class_to_idx, idx_to_class = self._find_classes(self.root)
        samples = make_dataset(self.root, class_to_idx, os.path.join(self.root, '..', self.split + '.txt'))

        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root))

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.idx_to_class = idx_to_class
        self.samples = samples
        self.targets = [s[1] for s in samples]

    def get_indexes_by_class(self):
        labels = {}
        for i,t in enumerate(self.targets):
            if t not in labels:
                labels[t] = []
            labels[t].append(i)
        return labels

      def get_class_name(self, i):
        return idx_to_class[i]

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.
        Args:
            dir (string): Root directory path.
        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.
        Ensures:
            No class is a subdirectory of another.
        """
        classes = [d.name for d in os.scandir(dir) if d.is_dir() and not d.name.lower().startswith('background')]
        classes.sort()
        for i in range(len(classes)):
          class_to_idx = {classes[i]: i}
          idx_to_class = {i: classes[i]}
        return classes, class_to_idx, idx_to_class

    def __getitem__(self, index):
        '''
        __getitem__ should access an element through its index
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        '''

        image, label = self.samples[index]  # Provide a way to access image and label via index
                                            # Image should be a PIL Image
                                            # label can be int

        # Applies preprocessing when accessing the image
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        '''
        The __len__ method returns the length of the dataset
        It is mandatory, as this is used by several other components
        '''
        length = len(self.samples)  # Provide a way to get the length (number of elements) of the dataset
        return length