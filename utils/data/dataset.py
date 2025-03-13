from torch.utils.data import Dataset
import torch
import numpy as np


class PittAdDataset(Dataset):
    def __init__(self, train_data, images):
        super(PittAdDataset, self).__init__()
        self.images = images
        self.train_data = train_data

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        image_name = self.images.keys()[item]
        image = self.images[image_name]
        try:
            options = self.train_data[image][1]
            correct_answers = self.train_data[image][0]
        except:
            return None
        return image, options, correct_answers