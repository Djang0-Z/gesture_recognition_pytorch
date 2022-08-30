import numpy as np
import models
from CSIKit.reader import NEXBeamformReader
from CSIKit.reader import get_reader
from CSIKit.util import csitools
import torch
from torch.utils.data import Dataset
from torchvision import transforms

path='../data/test/user6-1-1-1-1-r2.dat'
#path='../data/test/user6-1-1-1-1-r5.dat'
#path='../data/test/user6-2-1-2-4-r4.dat'
#path='../data/test/user6-2-1-2-4-r4.dat'
#path='../data/test/user6-3-1-2-1-r5.dat'
#path='../data/test/user6-3-1-2-2-r1.dat'
#path='../data/test/user6-4-3-4-4-r4.dat'
#path='../data/test/user6-4-3-4-4-r5.dat'
#path='../data/test/user6-5-2-4-5-r2.dat'
#path='../data/test/user6-5-5-2-2-r4.dat'
#path='../data/test/user6-6-5-5-5-r2.dat'
#path='../data/test/user6-6-5-5-5-r4.dat'
model = models.CNN2()
model.eval()
with torch.no_grad():
    csi_data = get_reader(path).read_file(path)
    csi_matrix, _, _ = csitools.get_CSI(csi_data, metric='amplitude')
    csi_matrix = csi_matrix[:, :, :, :].T  #
    csi_matrix = csi_matrix.reshape(-1)
    temp_matrix = np.ones(128 * 150) * csi_matrix.mean()
    csi_matrix = np.concatenate([csi_matrix, temp_matrix], axis=0)[:128 * 150]
    csi_matrix = csi_matrix.reshape(1, 128, 150)  # 1, 128, 150
    csi_matrix = torch.from_numpy(csi_matrix).float()
    csi_matrix = csi_matrix.clamp(min=-20)
    transform = transforms.RandomAffine(0, translate=(0.3, 0))
    csi_matrix=transform(csi_matrix)
    csi_matrix=torch.reshape(csi_matrix,(1,1,128,150))
    model.load_state_dict(torch.load("../src/models/model-1.pkl"))
    output=model(csi_matrix)
    print(output)
    print(output.argmax(1))