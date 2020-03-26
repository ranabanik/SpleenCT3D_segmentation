# rana.banik@vanderbilt.edu
# works for non-overlapping tiles: with step size 64, 128, 256
# and image size multiples of 64 in coronal and saggital
"""pseudocode
    for tl -> #allTiles:
        if tl divisible by #TilesPerSlice:
            axial increases 1
            coronal, sagittal = 0
        else
            if tl divisible by (#TilesinCorDirection * #TilesinSagDirecton)
                coronal increases by step
            else
                saggital increases by step
    """
import os
import nibabel as nib
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import h5py

if __name__ != '__main__':
    imgDir = r'C:\StorageDriveAll\Data\Assignment3\Testing\img'
    imgPath = glob(os.path.join(imgDir, '*.gz'))
    # print(imgPath)
    hpath = os.path.join(imgDir, 'SpleenTest2DTiles.h5')
    demoImg = nib.load(imgPath[8]).get_data()
    imgDim = np.shape(demoImg)
    # print(imgDim, imgDim[0])
    step = 128
    nTilesPerSlice = np.int((imgDim[0] / step) * (imgDim[1] / step))

# print(nTilesPerSlice)
if __name__ != '__main__':
    with h5py.File(hpath, 'w') as f:
        for i in imgPath:
            brokenCT = []
            ct = nib.load(i).get_data()
            print(ct.shape, '\n')
            for axi in range(ct.shape[2]):
                for cor in range(0, ct.shape[0], step):
                    for sag in range(0, ct.shape[1], step):
                        # print(cor, sag, axi)
                        img = ct[cor:cor + step, sag:sag + step, axi]
                        brokenCT.append(img)

            ID = i[48:-7]
            f[ID] = brokenCT

if __name__ != '__main__': # comment out this line if you want to see the stitching
    brokenCT = np.array(brokenCT)
    stichCT = np.zeros_like(ct)
    """this portions stiches the tiles back into 3D image"""
    sag = 0
    cor = 0
    axi = 0
    for tl in range(np.shape(brokenCT)[0]):  # 552
        slice = brokenCT[tl, :, :]
        stichCT[cor:cor + step, sag:sag + step, axi] = slice
        if (tl + 1) % nTilesPerSlice == 0:
            axi += 1
            cor = 0
            sag = 0
            if __name__ != '__main__':
                """to see every slice"""
                fig, ax = plt.subplots(np.int(imgDim[0]/step), np.int((imgDim[1]/step)), figsize=(20, 20))
                axes = ax.ravel()
                for ind, ii in enumerate(range(tl, tl+nTilesPerSlice)):
                    axes[ind].imshow(brokenCT[ii, :, :], cmap='gray')
                    axes[ind].axis('off')
                fig.tight_layout()
                fig.show()
                # ind = 0
        else:
            if (tl + 1) % np.int(nTilesPerSlice / (imgDim[0]/step)) == 0:
                cor += step
                sag = 0
            else:
                sag += step
        if __name__ == '__main__':
            """to see only the mid axial slice"""
            if axi == np.int(ct.shape[2]/2):
                fig, ax = plt.subplots(np.int((imgDim[0]/step)), np.int(imgDim[0]/step), figsize=(20, 20))
                axes = ax.ravel()
                for ind, i in enumerate(range(tl, tl+nTilesPerSlice)):
                    axes[ind].imshow(brokenCT[i, :, :], cmap='gray')
                    axes[ind].axis('off')
                fig.tight_layout()
                fig.show()
    if __name__ != '__main__':
        axial = 67
        plt.imshow(stichCT[:, :, axial], cmap='gray')
        plt.show()

    # break

# """check the h5py file"""
# f = h5py.File(hpath, 'r')
# # print(f.keys())
# CT = f['img0080']
# print(np.shape(CT))

if __name__ == '__main__':
    trainDir = r'/media/banikr2/DATA/SpleenCTSegmentation/Training'
    imgDir = os.path.join(trainDir, 'img')
    labDir = os.path.join(trainDir, 'splabel')
    trainFileList = r'/media/banikr2/DATA/SpleenCTSegmentation/Training/TrainList.txt'
    with open(os.path.join(trainFileList), 'r') as f:
        filenames = f.readlines()
    filenames = [item.strip() for item in filenames]
    # print(filenames)  # 30
    subCT = []
    subMask = []
    imgDim = [512, 512]
    step = 256
    nTilesPerSlice = np.int((imgDim[0] / step) * (imgDim[1] / step))
    traindataPath = os.path.join(trainDir, '2D512NonOverLapTrain.h5')
    validdataPath = os.path.join(trainDir, '2D512NonOverLapValid.h5')
    ind = 0
    with h5py.File(traindataPath, 'w') as f:
        for fInd in filenames:
            print(fInd[:-7])
            ind += 1
            if ind > 3:
            # subCT = []
            # subMask = []
                ct = nib.load(os.path.join(imgDir, os.path.basename(imgDir) + fInd)).get_data()
                #     print(ct.shape)
                # totSlice += ct.shape[2]
                mask = nib.load(os.path.join(labDir, 'mask' + fInd)).get_data()
                for axi in range(ct.shape[2]):
                    # print(cor, sag, axi)
                    img = ct[:, :, axi]
                    subCT.append(img)
                    # print(np.shape(subCT))
                    msk = mask[:, :, axi]
                    subMask.append(msk)
                # break
        f['img'] = subCT
        f['msk'] = subMask

