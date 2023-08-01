from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import torch
import pickle as pkl
from torch.utils.data import Dataset
         
# 训练 office source-free model所用data
class officeDataset_target(Dataset):
    def __init__(self, root, label_file, train=True, transform=None):
        super(officeDataset_target, self).__init__()
        self.root = root
        self.transform = transform
        self.train = train  # training set or test set
        self.label_file = label_file
        with open(self.label_file, 'rb') as f:
            train_dict = pkl.load(f)
        train_list = train_dict['train_list']

        # now load the picked numpy arrays
        if self.train:
            self.train_data = []
            self.train_labels = []
            for i in range(len(train_list)):
                self.train_data.append(os.path.join(self.root, train_list[i][0]))
                self.train_labels.append(train_list[i][1])

        else:
            self.test_data = []
            self.test_labels = []
            for i in range(len(train_list)):
                self.test_data.append(os.path.join(self.root, train_list[i][0]))
                self.test_labels.append(train_list[i][1])

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img_name, target = self.train_data[index], self.train_labels[index]
        else:
            img_name, target = self.test_data[index], self.test_labels[index]

        img = Image.open(img_name).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, target, index

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)


# 训练 source model 所用 data
class visDataset(Dataset):
    """
    ASOCT-2 class
    """

    def __init__(self, root, label_file, train=True, transform=None):
        super(visDataset, self).__init__()
        self.root = root
        self.transform = transform
        self.train = train  # training set or test set
        self.label_file = label_file
        with open(self.label_file, 'rb') as f:
            train_dict = pkl.load(f)
            # print(train_dict,type(train_dict))
        train_list = train_dict['train_list']
        val_list = train_dict['test_list']

        # now load the picked numpy arrays
        if self.train:
            self.train_data = []
            self.train_labels = []
            for i in range(len(train_list)):
                self.train_data.append(os.path.join(self.root, train_list[i][0]))
                self.train_labels.append(train_list[i][1])

        else:
            self.test_data = []
            self.test_labels = []
            for i in range(len(val_list)):
                self.test_data.append(os.path.join(self.root, val_list[i][0]))
                self.test_labels.append(val_list[i][1])

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img_name, target = self.train_data[index], self.train_labels[index]
            target = np.array(target).astype(int)
            target = torch.from_numpy(target)
        else:
            img_name, target = self.test_data[index], self.test_labels[index]
            target = np.array(target).astype(int)
            target = torch.from_numpy(target)

        img = Image.open(img_name).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, target, index

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)


# 训练 visda source-free model 所用 data
class visDataset_target(Dataset):
    def __init__(self, root, label_file, train=True, transform=None):
        super(visDataset_target, self).__init__()
        self.root = root
        self.transform = transform
        self.train = train  # training set or test set
        self.label_file = label_file
        with open(self.label_file, 'rb') as f:
            train_dict = pkl.load(f)
        train_list = train_dict['train_list']

        # now load the picked numpy arrays
        if self.train:
            self.train_data = []
            self.train_labels = []
            for i in range(len(train_list)):
                self.train_data.append(os.path.join(self.root, train_list[i][0]))
                self.train_labels.append(train_list[i][1])

        else:
            self.test_data = []
            self.test_labels = []
            for i in range(len(train_list)):
                self.test_data.append(os.path.join(self.root, train_list[i][0]))
                self.test_labels.append(train_list[i][1])

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img_name, target = self.train_data[index], self.train_labels[index]
        else:
            img_name, target = self.test_data[index], self.test_labels[index]

        img = Image.open(img_name).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, target, index

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)


class visDataset_target_select(Dataset):
    def __init__(self, root, label_file, pseudo_label, select_index, transform=None):
        super(visDataset_target_select, self).__init__()
        self.root = root
        self.transform = transform
        self.label_file = label_file
        with open(self.label_file, 'rb') as f:
            train_dict = pkl.load(f)
        train_list = train_dict['train_list']

        self.train_data = []
        self.train_labels = []
        for i in range(len(train_list)):
            if i in select_index:
                self.train_data.append(os.path.join(self.root, train_list[i][0]))
                self.train_labels.append(pseudo_label[i])

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img_name, target = self.train_data[index], self.train_labels[index]

        img = Image.open(img_name).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, target, index

    def __len__(self):
        return len(self.train_data)
