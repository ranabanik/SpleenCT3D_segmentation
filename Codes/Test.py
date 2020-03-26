import torch
from torch.utils import data
from Utility import SpleenDataset, getLargestCC, calculate_nifti_all_labels_dice_score
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
segDir = r'/media/banikr2/DATA/SpleenCTSegmentation/Testing/Segmentation'
trainResult = r'/media/banikr2/DATA/SpleenCTSegmentation/Training/result'
trainFileList = r'/media/banikr2/DATA/SpleenCTSegmentation/Training/TrainList.txt'
with open(os.path.join(trainFileList), 'r') as f:
    filenames = f.readlines()
filenames = [item.strip() for item in filenames]

if __name__ != '__main__':
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    TIME_STAMP = '2020-03-25-01-19-13'

    Model = UNet(in_dim=1, out_dim=1, num_filters=32).to(device) #64 for 128x128
    FILEPATH_MODEL_LOAD = os.path.join(modelDir, '{}.pt'.format(TIME_STAMP))
    train_states = torch.load(FILEPATH_MODEL_LOAD)
    Model.load_state_dict(train_states['train_states_best']['model_state_dict'])

if __name__ != '__main__':
    filenames = glob(os.path.join(dataDir, '*.gz'))
    ind = 0
    for fInd in filenames:
        ind += 1
        print(os.path.basename(fInd)[:-7])#fInd[56:-7]) #53 for test
        nib_obj = nib.load(fInd)
        img = nib_obj.get_data()
        nSlices = img.shape[2]
        # print(img.shape)
        # slices = np.shape(img)[2]
        mask = np.zeros(img.shape).astype('float32')
        img = np.transpose(img, (2, 0, 1)) # slicex512x512
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
        mask = getLargestCC(mask)
        print(mask.sum())
        mask = nib.Nifti1Image(mask.astype('uint8'), nib_obj.affine, nib_obj.header)
        savepath = os.path.join(trainResult, '{}.nii.gz'.format(os.path.basename(fInd)[:-7]))
        nib.save(mask, savepath)
        # if ind == 3:
        #     break

if __name__ == '__main__':
    # i = 2
    maskDir = r'/media/banikr2/DATA/SpleenCTSegmentation/Training/splabel'
    # resultDir = r'/media/banikr2/DATA/SpleenCTSegmentation/Testing'
    diceList = np.array([])
    for fInd in filenames:
        # i = os.path.basename(fInd[3:-7])
        # print(fInd)
        maskpath = os.path.join(maskDir, 'mask{}.nii.gz'.format(fInd[:-7]))
        mask = nib.load(maskpath).get_data()
        # print(mask.shape, mask.dtype, mask.sum())
        resultpath = os.path.join(trainResult, 'img{}.nii.gz'.format(fInd[:-7]))
        result = nib.load(resultpath).get_data().astype('uint8')
        # print(result.shape, result.dtype, result.max(), result.min(), result.sum())
        # getDSC = DiceLoss()
        commonArea = (result * mask)
        # print(commonArea.shape)
        dsc = (2*commonArea.sum())/(result.sum()+mask.sum())
        diceList = np.append(diceList, dsc)
        print("Dice score", fInd, ":", dsc)
        
    print(np.mean(diceList))
    print(np.median(diceList))
    print(np.std(diceList))

