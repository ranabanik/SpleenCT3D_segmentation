# rana.banik@vanderbilt.edu
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

imgDir = r'C:\StorageDriveAll\Data\Assignment3\Testing\img'
imgPath = glob(os.path.join(imgDir, '*.gz'))
# print(imgPath)

demoImg = nib.load(imgPath[8]).get_data()
imgDim = np.shape(demoImg)
# print(imgDim, imgDim[0])
step = 256
nTilesPerSlice = np.int((imgDim[0] / step) * (imgDim[1] / step))
# print(nTilesPerSlice)
for i in imgPath:
    brokenCT = []
    ct = nib.load(i).get_data()
    print(ct.shape, '\n')
    for axi in range(ct.shape[2]):
        for cor in range(0, ct.shape[0], step):
            for sag in range(0, ct.shape[1], step):
                print(cor, sag, axi)
                img = ct[cor:cor + step, sag:sag + step, axi]
                brokenCT.append(img)

    # print(np.shape(img)) #(256, 256)
    # print(np.shape(brokenCT)) #(552, 256, 256)
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

