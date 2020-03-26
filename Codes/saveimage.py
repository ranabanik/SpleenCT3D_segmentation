import os
import nibabel as nib
import numpy as np

trainDir = r'/media/banikr2/DATA/SpleenCTSegmentation/Training'
trainFileList = r'/media/banikr2/DATA/SpleenCTSegmentation/Training/TrainList.txt'
imgDir = os.path.join(trainDir, 'img')
labDir = os.path.join(trainDir, 'label')
splabDir = os.path.join(trainDir, 'splabel')
# os.mkdir(splabDir)
with open(os.path.join(trainFileList), 'r') as f:
    filenames = f.readlines()
filenames = [item.strip() for item in filenames]
# print(filenames) #30

for fInd in filenames:
    print(fInd[:-7])
    nib_obj = nib.load(os.path.join(labDir, os.path.basename(labDir) + fInd))
    mask = nib_obj.get_data()
    print(mask.dtype) # uint8
    print(mask.shape[2])
    # mask[np.where(mask != 1)] = 0
    # mask = nib.Nifti1Image(mask, nib_obj.affine, nib_obj.header)
    # savepath = os.path.join(splabDir, 'mask{}.nii.gz'.format(fInd[:-7]))
    # nib.save(mask, savepath)
