from torch.utils.data import Dataset
import data.util_2D as Util
import os
import numpy as np
from skimage import io
from natsort import natsorted

class MYDataset(Dataset):

    def __init__(self, dataroot, split='train'):
        self.split = split
        self.imageNum = []
        self.dataroot = dataroot

        # List all subfolders and sort them naturally
        subfolders = natsorted([f for f in os.listdir(os.path.join(dataroot, split)) if os.path.isdir(os.path.join(dataroot, split, f))])

        for folder in subfolders:
            folder_path = os.path.join(dataroot, split, folder)
            dataFiles = natsorted(os.listdir(folder_path))

            # Create pairs of consecutive elements
            for i in range(len(dataFiles) - 10):
                pair = [os.path.join(folder, dataFiles[i]), os.path.join(folder, dataFiles[i + 10])]
                self.imageNum.append(pair)

        self.data_len = len(self.imageNum)

    def __len__(self):
        return self.data_len


    def __getitem__(self, index):
        fileInfo = self.imageNum[index]
        dataX, dataY = fileInfo[0], fileInfo[1]
        dataXPath = os.path.join(self.dataroot, self.split, dataX)
        dataYPath = os.path.join(self.dataroot, self.split, dataY)
        data = io.imread(dataXPath, as_gray=True).astype(float)[:, :, np.newaxis]
        label = io.imread(dataYPath, as_gray=True).astype(float)[:, :, np.newaxis]

        dataX_RGB = io.imread(dataXPath).astype(float)
        dataY_RGB = io.imread(dataYPath).astype(float)

        [data, label] = Util.transform_augment([data, label], split=self.split, min_max=(-1, 1))

        return {'M': data, 'F': label, 'MC': dataX_RGB, 'FC': dataY_RGB, 'nS': 7, 'P': fileInfo, 'Index': index}