import torch
import torch.nn as nn
from torch.utils import data
from Utility import SpleenDataset, dice_loss
from model2D import UNet
import os
import h5py
import matplotlib.pyplot as plt
import numpy as np

dataDir = r'/media/banikr2/DATA/SpleenCTSegmentation'
modelDir = r'/media/banikr2/DATA/banikr_D_drive/model'

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

TIME_STAMP = '2020-03-23-02-53-11'

Model = UNet(in_dim=1, out_dim=1, num_filters=64).to(device)
FILEPATH_MODEL_LOAD = os.path.join(modelDir, '{}.pt'.format(TIME_STAMP))
train_states = torch.load(FILEPATH_MODEL_LOAD)
Model.load_state_dict(train_states['train_states_best']['model_state_dict'])

dataPath = os.path.join(dataDir, 'SpleenTest2DTiles.h5')
f = h5py.File(dataPath,'r')
# print(f.keys())
CT = f['img0070']
print(np.shape(CT))

testSet = SpleenDataset(CT, None)
# img = next(iter(testSet))
testLoader = data.DataLoader(testSet, batch_size=16, shuffle=False, num_workers=4)
img = next(iter(testLoader))
# print(img.shape)
# print(img.max(), img.min()) #tensor(254.8711, dtype=torch.float64) tensor(0.0919, dtype=torch.float64)
ind = 0
with torch.no_grad():
    for vbatch_idx, sample in enumerate(testLoader):
        ct = sample.float().to(device)
        out = Model(ct)
        # print(out.shape)
        ind += 1
        if ind == 50:
            out = out.detach().cpu().numpy().squeeze(1)
            # thr = 0.9
            # out[out > thr] = 1
            # out[out <= thr] = 0
            fig, ax = plt.subplots(4, 4, figsize=(20, 20))
            axes = ax.ravel()
            for ii in range(16):
                img = out[ii, :, :]
                print(img.max(), img.min(), img.std())
                axes[ii].imshow(img, cmap='gray')
                axes[ii].axis('off')
            fig.tight_layout()
            fig.show()


# CT_synth = pred.detach().cpu().numpy().squeeze(1)