import numpy as np
import torch
from random import gauss
from torch.utils import data

np.random.seed(101)

def get_paired_patch(mr, mr_patch_size, mask, mask_patch_size, nPatches) -> 'List':  # MR, patchsize
    """
    :param mr: MR image 3D
    :param inputpatchsize: [32,32,3]
    :param mask: Binary mask 3D
    :param maskpatchsize: [16,16,3]
    :param nPatches: how many patches
    :return: center shapes are e.g.: (3,500) format
    """
    # trainLabelDir = r'C:\StorageDriveAll\Data\Assignment3\TrainImageLabels'
    foreground = np.array(np.where(mask == 1))  #foreground shape-> (3, 11133) shape[1]-> take all points in mask gt 0
    # print(foreground.shape)
    centers = foreground[:, np.random.permutation(foreground.shape[1])[:nPatches]]
    # print(centers[:, -3])
    # mr_patch_size = inputpatchsize  # [32,32,3]
    # mask_patch_size = maskpatchsize  # [16,16,3]
    # print("Centers shape:", centers.shape)
    mr_patch = []
    mask_patch = []
    bottom = 0
    top = 0
    mid = 0
    for i in range(centers.shape[1]): # number of centers
        if centers[0, i] < int(mr_patch_size[0]/2): # 64
            centers[0, i] = int(mr_patch_size[0]/2)
            # top += 1
        elif centers[0, i] > int(mr.shape[0] - mr_patch_size[0]/2):
            centers[0, i] = int(mr.shape[0] - mr_patch_size[0]/2)

        if centers[1, i] < int(mr_patch_size[1] / 2):
            centers[1, i] = int(mr_patch_size[1] / 2)
            # top += 1
        elif centers[1, i] > int(mr.shape[1] - mr_patch_size[1] / 2):
            centers[1, i] = int(mr.shape[1] - mr_patch_size[1] / 2)

        if centers[2, i] < int(mr_patch_size[2] / 2):
            centers[2, i] = int(mr_patch_size[2] / 2)
            # top += 1
        elif centers[2, i] > int(mr.shape[2] - mr_patch_size[2] / 2):
            centers[2, i] = int(mr.shape[2] - mr_patch_size[2] / 2)
            # print("Bottom")
        #     bottom += 1
        # else:
        #     mid += 1
        #     pass #continue #if continue the loop starts from i again
            # print("Passed")
        # print(centers[:, i])
        patch1 = mr[(centers[0, i] - np.int(mr_patch_size[0] / 2)):(centers[0, i] + np.int(mr_patch_size[0] / 2)),
                (centers[1, i] - np.int(mr_patch_size[1] / 2)):(centers[1, i] + np.int(mr_patch_size[1] / 2)),
                (centers[2, i] - np.int(mr_patch_size[2] / 2)):(centers[2, i] + np.int(mr_patch_size[2] / 2))]
        # print(np.shape(patch1))
        mr_patch.append(patch1)

        patch2 = mask[(centers[0, i] - np.int(mask_patch_size[0] / 2)):(centers[0, i] + np.int(mask_patch_size[0] / 2)),
                (centers[1, i] - np.int(mask_patch_size[1] / 2)):(centers[1, i] + np.int(mask_patch_size[1] / 2)),
                (centers[2, i] - np.int(mask_patch_size[2] / 2)):(centers[2, i] + np.int(mask_patch_size[2] / 2))]
        # print(np.array(patch).shape)
        # print(np.shape(patch2))
        mask_patch.append(patch2)

        if (np.shape(patch1) != mr_patch_size) or (np.shape(patch2) != mask_patch_size):
            # print(np.shape(ctPatch))
            # print(fInd)
            print(i)
            msg = 'issues found in center: ', centers[:, i]
            # raise msg
            # break
            print(msg)
    # print("Top", top)
    # print("Bottom:", bottom)
    # print("Mid", mid)
    return mr_patch, mask_patch  # centers

def padzero(vector, pad_width, iaxis, kwargs):
    vector[:pad_width[0]] = 0
    vector[-pad_width[1]:] = 0
    return vector

class SpleenDataset(data.Dataset):
    def __init__(self, CT, Mask):  # todo: how many subMR should be in one input?? shape?
        self.CT = CT
        self.Mask = Mask

    def getSWN(self, img, x, y):  # soft window normalization
        #         x = 10;y=10
        L = gauss(40, x)
        W = gauss(200, y)
        #         print(L, W)
        W = abs(W)
        maxThr = L + W
        minThr = L - W
        img[img > maxThr] = maxThr
        img[img < minThr] = minThr
        img = 255 * ((img - minThr) / (maxThr - minThr))
        return img

    def __len__(self):
        return len(self.CT)

    def __getitem__(self, item):
        #         CT = self.CT[item]
        CT = self.getSWN(self.CT[item], 10, 10)
        CT = torch.tensor(CT)
        CT = CT.unsqueeze(0)  # (64, 64, 8) -> (1, 64, 64, 8) creating channel/filter
        Mask = self.Mask[item]
        Mask = torch.tensor(Mask)
        Mask = Mask.unsqueeze(0)
        return CT, Mask

def get_paired_patch2D(img, imgSize, msk, mskSize, nPatches):
    """
    :param img:
    :param imgSize:
    :param msk:
    :param mskSize:
    :param nPatches:
    :return:
    """
    foreground = np.array(np.where(msk == 1))
    centers = foreground[:, np.random.permutation(foreground.shape[1])[:nPatches]]
    img_patch = []
    msk_patch = []

    for i in range(centers.shape[1]): # number of centers
        if centers[0, i] < int(imgSize[0]/2): # 64
            centers[0, i] = int(imgSize[0]/2)
            # top += 1
        elif centers[0, i] > int(img.shape[0] - imgSize[0]/2):
            centers[0, i] = int(img.shape[0] - imgSize[0]/2)

        if centers[1, i] < int(imgSize[1] / 2):
            centers[1, i] = int(imgSize[1] / 2)
            # top += 1
        elif centers[1, i] > int(img.shape[1] - imgSize[1] / 2):
            centers[1, i] = int(img.shape[1] - imgSize[1] / 2)

        if centers[2, i] == img.shape[2]:
            centers[2,i] = img.shape[2] - 1
        elif centers[2, i] == 0:
            centers[2, i] = 1

        patch1 = img[(centers[0, i] - np.int(imgSize[0] / 2)):(centers[0, i] + np.int(imgSize[0] / 2)),
                 (centers[1, i] - np.int(imgSize[1] / 2)):(centers[1, i] + np.int(imgSize[1] / 2)),
                 (centers[2, i] - 1):(centers[2, i] + 2)]
        # print(np.shape(patch1))
        img_patch.append(patch1)

        patch2 = msk[(centers[0, i] - np.int(mskSize[0] / 2)):(centers[0, i] + np.int(mskSize[0] / 2)),
                 (centers[1, i] - np.int(mskSize[1] / 2)):(centers[1, i] + np.int(mskSize[1] / 2)),
                 (centers[2, i] - 1):(centers[2, i] + 2)]
        # print(np.array(patch).shape)
        # print(np.shape(patch2))
        msk_patch.append(patch2)

    return img_patch, msk_patch
