import glob
import os
import numpy as np
from CSIKit.reader import NEXBeamformReader
from CSIKit.reader import get_reader
from CSIKit.util import csitools
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class CsiDataSet(Dataset):
    def __init__(self, root='../data', files=None):

        self.root = root
        self.file = []
        self.label = []
        self.reader = get_reader
        self.labels = [ '1', '2', '3', '4', '5','6']
        self.transform = None

        if files is None:
            for up_path, dirs, file_names in os.walk(self.root, topdown=False):
                for file_name in file_names:
                    file_name = file_name.strip()
                    if not file_name:
                        continue
                    file_path = os.path.join(up_path, file_name)
                    self.file.append(file_path)
                    self.label.append(int(file_name.split('-')[1])-1)

                self.transform = transforms.RandomAffine(0, translate=(0.3, 0))
        else:
            if type(files) is list:
                self.file.extend(files)
            else:
                self.file.append(files)


    def __len__(self):
        return len(self.file)

    def __getitem__(self, index):
        file = self.file[index]

        csi_data = self.reader(file).read_file(file)
        csi_matrix, _, _ = csitools.get_CSI(csi_data, metric='amplitude')
        csi_matrix = csi_matrix[:, :, :, :].T
        csi_matrix = csi_matrix.reshape(-1)
        temp_matrix = np.ones(128 * 150) * csi_matrix.mean()
        csi_matrix = np.concatenate([csi_matrix, temp_matrix], axis=0)[:128 * 150]
        csi_matrix = csi_matrix.reshape(1, 128, 150) #1, 128, 150
        csi_matrix = torch.from_numpy(csi_matrix).float()
        csi_matrix = csi_matrix.clamp(min=-20)

        if self.transform is not None:
            csi_matrix = self.transform(csi_matrix)

        if self.label:
            return csi_matrix, self.label[index]
        else:
            return csi_matrix, -1

    def get_label(self, idx):
        return self.labels[idx]


