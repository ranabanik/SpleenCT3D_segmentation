import torch
from torch.utils import data
from Utility import SpleenDataset, getLargestCC
from model2D import UNet
import os
import numpy as np
import nibabel as nib
from glob import glob
import cv2

# dataDir = r'/media/banikr2/DATA/SpleenCTSegmentation' # 128x128
dataDir = r'/media/banikr2/DATA/SpleenCTSegmentation/Training/img' #512x512
modelDir = r'/media/banikr2/DATA/banikr_D_drive/model'
testDir = r'/media/banikr2/DATA/SpleenCTSegmentation/Testing/img'
resultDir = r'/media/banikr2/DATA/SpleenCTSegmentation/Testing/result'

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

TIME_STAMP = '2020-03-25-01-19-13'

Model = UNet(in_dim=1, out_dim=1, num_filters=32).to(device) #64 for 128x128
FILEPATH_MODEL_LOAD = os.path.join(modelDir, '{}.pt'.format(TIME_STAMP))
train_states = torch.load(FILEPATH_MODEL_LOAD)
Model.load_state_dict(train_states['train_states_best']['model_state_dict'])
# dataPath = os.path.join(dataDir, 'Valid512_03.h5')
# f = h5py.File(dataPath, 'r')
# # print(f.keys())
# CT = f['img']
# print(np.shape(CT))
# testSet = SpleenDataset(CT, None)
# # img = next(iter(testSet))
# testLoader = data.DataLoader(testSet, batch_size=16, shuffle=False, num_workers=4)
# img = next(iter(testLoader))
# print(img.shape)
# # print(img.max(), img.min()) #tensor(254.8711, dtype=torch.float64) tensor(0.0919, dtype=torch.float64)
# ind = 0
# if __name__ == '__main__':
#     """for 128 x 128"""
#     with torch.no_grad():
#         for vbatch_idx, sample in enumerate(testLoader):
#             ct = sample.float().to(device)
#             out = Model(ct)
#             # print(out.shape)
#             # ind += 1
#             # if ind == 6:
#             #     out = out.detach().cpu().numpy().squeeze(1)
#             #     ct = ct.detach().cpu().numpy().squeeze(1)
#             #     print(out.shape)
#             #     # thr = 0.99
#             #     # out[out > thr] = 1
#             #     # out[out <= thr] = 0
#             #     fig, ax = plt.subplots(4, 1, figsize=(30, 20))
#             #     axes = ax.ravel()
#             #     # for ii in range(16):
#             #     # img = out[ii, :, :]
#             #     ind = 0
#             #     for i in range(4):
#             #         axes[i].imshow(ct[ind], cmap='gray', alpha=0.3)
#             #         # print(img.max(), img.min(), img.std())
#             #         axes[i].imshow(out[ind], cmap='gray')
#             #         ind += 1
#             #     # axes.axis('off')
#             #     # fig.tight_layout()
#             #     # fig.show()
#             #     plt.show()

# CT_synth = pred.detach().cpu().numpy().squeeze(1)
if __name__ == '__main__':
    filenames = glob(os.path.join(dataDir, '*.gz'))
    ind = 0
    for fInd in filenames:
        ind += 1
        print(fInd[54:-7]) #53 for test
        nib_obj = nib.load(fInd)
        img = nib_obj.get_data()
        nSlices = img.shape[2]
        # print(img.shape)
        # slices = np.shape(img)[2]
        mask = np.zeros(img.shape).astype('float32')
        img = np.transpose(img, (2, 0, 1)) # slicex512x512
        # print(img.shape)
        # for axi in range(img.shape[2]):
        #     # print(cor, sag, axi)
        #     slice = ct[:, :, axi]
        #
        # n = 0
        testSet = SpleenDataset(img, None)
        tBatchSize = 20
        testLoader = data.DataLoader(testSet, batch_size=tBatchSize, shuffle=False, num_workers=4, drop_last=False)
        with torch.no_grad():
            for vbatch_idx, sample in enumerate(testLoader):
                # print(vbatch_idx)
                img = sample.float().to(device)
                out = Model(img)
                out = out.detach().cpu().numpy().squeeze(1)#.astype('uint8')
                out = np.transpose(out, (1, 2, 0)) # (512, 512, 10)
                # print(type(out), out.dtype, out.max()) # float32 0.99976856
                # print(out.shape)
                if vbatch_idx == np.ceil(nSlices/tBatchSize) - 1:
                    print('here')
                    mask[:, :, vbatch_idx*tBatchSize:nSlices] = out
                else:
                    mask[:, :, vbatch_idx*out.shape[2]:(1+vbatch_idx)*out.shape[2]] = out#*255
                # mask[:, ]
                # n += 1
        maxpix = mask.max()
        # mask = np.array(mask)
        print(maxpix)
        mask[mask >= (maxpix/2)] = 1
        mask[mask < (maxpix/2)] = 0
        print(mask.sum())
        """Morphological operation"""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)) #doesn't work with 3D kernel
        mask = cv2.dilate(mask, kernel, iterations=2)
        # contour, hier = cv2.findContours(mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        # mask, _ = cv2.findContours(mask.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # mask = max(mask, key=cv2.contourArea)
        # mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)
        # mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)
        mask = getLargestCC(mask)
        print(mask.sum())
        mask = nib.Nifti1Image(mask.astype('uint8'), nib_obj.affine, nib_obj.header)
        savepath = os.path.join(resultDir, 'larcomp2{}.nii.gz'.format(fInd[54:-7]))
        nib.save(mask, savepath)
        if ind == 3:
            break