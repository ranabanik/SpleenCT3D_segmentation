"""
rana.banik@vanderbilt.edu
filepath should have the following structure.
C:.
├───Testing
│   └───img
└───Training
    ├───img
    └───label

All the image and label sizes are the same

Coronal range: 512 to 512
Sagittal range: 512 to 512
Axial range: 85 to 198

Train image intensity ranges from -3024 to 3095. Some of the images are from -1024, those have round
shape backgrounds, all the lowest intensities come from the background.
In Hounsfield unit, less than -1000 is air and greater than 2000 is metal. Cortical bones are 1800-1900 range
shouldn't be present in abdominal data.
Train label intensity ranges from 0 to 13, with spleen being 1 and background = 0

0006.nii.gz has the minimum 43233 pixels of spleen volume.
0009.nii.gz with 1294971 pixels of spleen volume is the largest.

Axially slices are more near to the end of the images like 5 slices in some cases.

Maximum intensity 3071  0037.nii.gz(index 26)  (-950)
Minimum intensity -1024

0001.nii.gz 1665 -1024
0002.nii.gz 879 -1024
0003.nii.gz 1038 -1024
0004.nii.gz 1016 -1018
0005.nii.gz 933 -1022
0006.nii.gz 1218 -1020
0007.nii.gz 964 -1011
0008.nii.gz 1090 -1024
0009.nii.gz 1126 -1006
0010.nii.gz 903 -835
0021.nii.gz 840 -1015
0022.nii.gz 1009 -1024
0023.nii.gz 1184 -1024
0024.nii.gz 949 -970
0025.nii.gz 993 -976
0026.nii.gz 1227 -1024
0027.nii.gz 759 -1008
0028.nii.gz 1006 -1024
0029.nii.gz 896 -1024
0030.nii.gz 918 -1024
0031.nii.gz 809 -1024
0032.nii.gz 802 -1007
0033.nii.gz 892 -939
0034.nii.gz 941 -992
0035.nii.gz 1106 -1010
0036.nii.gz 1015 -1024
0037.nii.gz 1984 -1024   **have the largest range...
0038.nii.gz 989 -1024
0039.nii.gz 988 -1024
0040.nii.gz 1092 -863

lowest spleen in number 10, total num of spleen voxels = 53869
"""
import os
from glob import glob
from tqdm import tqdm
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from Utility import get_paired_patch, get_paired_patch2D
import cv2  # for clustering
import h5py
trainDir = r'C:\StorageDriveAll\Data\Assignment3\Training'
trainImageDir = os.path.join(trainDir, 'img')
trainLabelDir = os.path.join(trainDir, 'label')
imagePath = glob(os.path.join(trainImageDir, '*.gz'))
labelPath = glob(os.path.join(trainLabelDir, '*.gz'))
trainFileList = r'C:\StorageDriveAll\Data\Assignment3\Training\TrainList.txt'
if __name__ != '__main__':
    with open(trainFileList, 'w') as file:
        for fl in imagePath:
            file.write('{}\n'.format(fl[64:]))
with open(os.path.join(trainFileList), 'r') as f:
    filenames = f.readlines()
filenames = [item.strip() for item in filenames]
print(len(filenames)) #30
if __name__ != '__main__':
    differentLabelImage = []
    howmanydiff = 0
    for i in filenames:
        # print(i)
        im = nib.load(os.path.join(trainImageDir, os.path.basename(trainImageDir) + i))
        lb = nib.load(os.path.join(trainLabelDir, os.path.basename(trainLabelDir) + i))
        # print(im.shape, lb.shape)#[0]==lb.shape[0] or im.shape[1] == lb.shape[1] or im.shape[2]==lb.shape[2])
        if im.shape != lb.shape:
            howmanydiff += 1
            differentLabelImage.append(i)
    if howmanydiff != 0:
        print(">>> Images and labels are not equal in size!!")
        print(differentLabelImage)
    else:
        print(">>> All the images and labels have same dimensions.")
if __name__ != '__main__':
    fileList = []
    x = []
    y = []
    z = []
    for path in imagePath:
        filename = os.path.basename(path)
        # print(filename)
        fileList.append(filename)
        im = nib.load(path).get_data()
        print(im.shape)
        x.append(im.shape[0])
        y.append(im.shape[1])
        z.append(im.shape[2])
    fig, ax = plt.subplots(3, 1, dpi=200)
    # ax = axes.ravel()
    ax[0].plot(x, color=(1, 0, 1), linewidth=1.5, label='X axis')
    ax[0].legend(loc='best')
    ax[0].grid()
    ax[1].plot(y, color=(0.9, 0.5, 0.2), linewidth=1.5, label='Y axis')
    ax[1].legend(loc='best')
    ax[1].grid()
    ax[2].plot(z, color=(0.4, 1, 0.1), linewidth=1.5, label='Z axis')
    ax[2].legend(loc='best')
    ax[2].grid()
    fig.suptitle('shape/#slices of axes', fontsize=10)
    fig.subplots_adjust(top=0.85)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    print('Coronal range:', min(x), 'to', max(x))  # 512 to 512
    print('Sagittal range:', min(y), 'to', max(y))  # 512 to 512
    print('Axial range:', min(z), 'to', max(z))  # 85 to 198
if __name__ != '__main__':
    maxTrainIntensity = -np.inf
    minTrainIntensity = np.inf
    for i in tqdm(filenames):
        # im = nib.load(os.path.join(trainImageDir, os.path.basename(trainImageDir) + i)).get_data()
        lb = nib.load(os.path.join(trainLabelDir, os.path.basename(trainLabelDir) + i)).get_data()
        tempMaxInt = lb.max() #im
        tempMinInt = lb.min()
        print("\n > {} ranges from {} to {}".format(i, tempMinInt, tempMaxInt))
        if tempMaxInt > maxTrainIntensity:
            maxTrainIntensity = tempMaxInt
        if tempMinInt < minTrainIntensity:
            minTrainIntensity = tempMinInt

    print(">>> Train image intensity ranges from {} to {}".format(minTrainIntensity, maxTrainIntensity))
if __name__ != '__main__':
    """Takes points from the mask"""
    lb = nib.load(os.path.join(trainLabelDir, os.path.basename(trainLabelDir) + filenames[3])).get_data()
    print(lb.shape)
    foreground = np.array(np.where(lb == 1))
    print(foreground.shape)
    nPatches = 5000
    centers = foreground[:, np.random.permutation(foreground.shape[1])[:nPatches]]
    # centers = get_paired_patch(lb, num_centers=5000)  # todo: change
    # print(centers[:, 23])
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(dpi=300)
    ax = fig.gca(projection='3d')
    ax.set_xlabel("coronal")
    ax.set_ylabel("saggital")
    ax.set_zlabel("axial")
    ax.grid(grid=True)
    ax.scatter(centers[0, :], centers[1, :], centers[2, :], c='r', marker='.')
    plt.title("Center points")
    ax.view_init(elev=60., azim=-165)
    plt.show()
    # sag = 50
    # cor = 50
    # axi = 3 #lowest 8
if __name__ != '__main__':
    """Can simple KNN algorithm classify well?"""
    # print(imagePath[23])
    image = nib.load(os.path.join(trainImageDir, os.path.basename(trainImageDir) + filenames[3])).get_data()
    # print(type(image)) #-> class 'numpy.ndarray'
    image = np.float32(image)
    # image = cv2.CV_32FC2(image)
    # print(type(image))
    vector_image = image.reshape((-1))
    # image.dtype = np.float329
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10000, 0.0002)
    k = 14
    _, labels, (centers) = cv2.kmeans(vector_image, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # centers = np.uint8(centers)
    # labels = labels.flatten()
    # centers shape: (5, 1)
    # labels shape: (655360, 1)
    segmented_image = centers[labels]  # .flatten()]
    segmented_image = segmented_image.reshape(image.shape)
    # print(centers.shape)#, labels)
    print(segmented_image.shape)
    fig, axes = plt.subplots(2, 5, figsize=(64, 20), squeeze=False)  # todo: change it based on image.shape[2]
    fig.subplots_adjust(top=1.5)
    ax = axes.ravel()
    for i in range(image.shape[2]):
        ax[i].imshow(segmented_image[:, :, i], cmap='Paired')  # , label='{}'.format(i+1))
        ax[i].set_ylabel('{}'.format(i + 1), fontsize=50, rotation=90)
        # ax[i].axis('Off')
        # .format(i+1), fontsize=40)
        # ax[i].legend(loc='upper right')
    fig.tight_layout()
    fig.suptitle("Axial slices of kmeans clustering", fontsize=50, y=1.0)
    # format_axes(fig)
    plt.show()
    """K means clustering do not perform well to segment the spleen from the MRI"""
# if __name__ == '__main__':
#     count = 0
#     for lb in labelPath:
#         # print(lb)
#         label = nib.load(lb).get_data()
#         if np.sum(label[:, :, label.shape[2]]) != 0:
#             print(lb)
#             count +=1
#     print(count)
"""condition-> Center comes from mask:
if patchCenter in top or bottom slice:
   #if odd number patchslice   
      if top:
        patchCenter coordinate-> X,Y,0+1 
      elif bottom:
        patchCenter coordinate-> X,Y,mask.shape[2]-1"""
if __name__ != '__main__':
    # fInd = filenames[23]
    # print(filenames[fInd])
    trainCT = []
    trainMask = []
    # ind = 0
    for fInd in filenames:
        print(fInd[:-7])
        ct = nib.load(os.path.join(trainImageDir, os.path.basename(trainImageDir) + fInd)).get_data()
    #     print(ct.shape)
        mask = nib.load(os.path.join(trainLabelDir, os.path.basename(trainLabelDir) + fInd)).get_data()
        mask[np.where(mask != 1)] = 0
        # print(mask.shape, "mask")
    #     # plt.imshow(mask[:,:,-1], cmap='gray')
    #     # plt.show()
    #     # print(mask.shape)
        ct_size = (64, 64, 8)
        mask_size = (32, 32, 8)
        ctPatch, maskPatch = get_paired_patch(ct, ct_size, mask, mask_size, 100)
        # print(np.shape(ctPatch))
        # print(np.shape(maskPatch))
        # if (np.shape(ctPatch) != ct_size) or (np.shape(maskPatch) != mask_size):
        #     print(np.shape(ctPatch))
        #     print(fInd)
        #     break
        trainCT.append(ctPatch); trainMask.append(maskPatch)
    # print("error free!")
        print("Train input shape:", np.shape(trainCT))
        print("Train mask shape:", np.shape(trainMask))
    #     # ind += 1
    #     # if ind == 2:
    #     #     break
    hpath = os.path.join(trainDir, 'SpleenData100.h5')
    with h5py.File(hpath, 'w') as f:
        f['img'] = trainCT
        f['mask'] = trainMask
    # print(mrPatch[0].shape, maskPatch[0].shape)
    # mrPatch = np.array(mrPatch); maskPatch = np.array(maskPatch)
    # fig, ax = plt.subplots(1, 2, figsize=(64, 20))
    # ax[0].imshow(mrPatch[258, :, :, 2], cmap='gray')
    # ax[1].imshow(maskPatch[258, :, :, 2], cmap='gray')
    # plt.show()

    # print(get_paired_patch.__annotations__)

if __name__ == '__main__':
    subCT = []
    subMask = []
    imgDim = [512, 512]
    step = 256
    nTilesPerSlice = np.int((imgDim[0] / step) * (imgDim[1] / step))
    hpath = os.path.join(trainDir, 'SpleenTilesNonOverLap.h5')
    totSlice = 0
    with h5py.File(hpath, 'w') as f:
        for fInd in filenames:
            print(fInd[:-7])
            # subCT = []
            # subMask = []
            ct = nib.load(os.path.join(trainImageDir, os.path.basename(trainImageDir) + fInd)).get_data()
            #     print(ct.shape)
            totSlice += ct.shape[2]
            mask = nib.load(os.path.join(trainLabelDir, os.path.basename(trainLabelDir) + fInd)).get_data()
            mask[np.where(mask != 1)] = 0
            for axi in range(ct.shape[2]):
                for cor in range(0, ct.shape[0], step):
                    for sag in range(0, ct.shape[1], step):
                        # print(cor, sag, axi)
                        img = ct[cor:cor + step, sag:sag + step, axi]
                        subCT.append(img)
                        # print(np.shape(subCT))
                        msk = mask[cor:cor + step, sag:sag + step, axi]
                        subMask.append(msk)
        print("Total number of slices:", totSlice)
                        # print(np.shape(subMask))
            # imgID = fInd[:-7]+'img'
            # mskID = fInd[:-7]+'msk'
        f['img'] = subCT
        f['msk'] = subMask

if __name__ != '__main__':
    trainPath = glob(os.path.join(trainDir, '*.h5'))[0]
    print(trainPath)
    f = h5py.File(trainPath, 'r')
    print(f.keys())

if __name__ != '__main__':
    """
    Smallest mask is: DET0028301_avg.nii.gz 
    having 2646 binary values. 
    Actual mask size: 192x156x8
    So for this mask
    number of patches should be less than 2646
       
    largest mask is DET0003501_avg.nii.gz 
    having 63894 binary pixels with 1.
    Actual mask size: 512x512x15
       
    Number of patches taken from masks/images should vary 
    """
    minval = np.inf
    maxval = -np.inf
    for fInd in filenames:
        mask = nib.load(os.path.join(trainLabelDir, os.path.basename(trainLabelDir) + fInd)).get_data()
        einPixTot = np.shape(np.where(mask == 1))[1]
        # einPixTot = mask.ravel().sum()
        if einPixTot < minval:
            print(fInd, einPixTot)
            minval = einPixTot

        # if einPixTot > maxval:
        #     print(fInd, einPixTot)
        #     maxval = einPixTot
    # print(minval)
if __name__ != '__main__':  #todo: find how the mask 1 pixels are spread
    corLabSpleen = []
    sagLabSpleen = []
    axiLabSpleen = []
    for fInd in filenames:
        print(fInd)
        mask = nib.load(os.path.join(trainLabelDir, os.path.basename(trainLabelDir) + fInd)).get_data()
        print(mask.shape)
        mask[np.where(mask != 1)] = 0
        start = 0
        end = mask.shape[0]
        for cor in range(mask.shape[0]):
            if np.sum(mask[cor, :, :]) != 0:
                start = cor
                break
        for cor in range(start, mask.shape[0]):
            if np.sum(mask[cor, :, :]) == 0:
               end = cor - 1
               break
        # break
        # print(start, end)
        corLabSpleen.append([start, end])
        print([start, end])
        start = 0
        end = mask.shape[1]
        for sag in range(mask.shape[1]):
            if np.sum(mask[:, sag, :]) != 0:
                start = sag
                break
        for sag in range(start, mask.shape[1]):
            if np.sum(mask[:, sag, :]) == 0:
               end = sag - 1
               break
        sagLabSpleen.append([start, end])
        print([start, end])
        start = 0
        end = mask.shape[2]
        for axi in range(mask.shape[2]):
            if np.sum(mask[:, :, axi]) != 0:
                start = axi
                break # this break from the for loop right above
        for axi in range(start, mask.shape[2]):
            if np.sum(mask[:, :, axi]) == 0:
                end = axi - 1
                break
        axiLabSpleen.append([start, end])
        print([start, end])
        # plt.imshow(mask[:, :, 146])
        # plt.show()
        # break

if __name__ != '__main__':
    """How changed the hdf format data and saved in the same file"""
    trainPath = glob(os.path.join(trainDir, '*.h5'))[0]
    f = h5py.File(trainPath, 'r+')
    CT = f['img']
    # print(type(CT))
    mask = f['mask']
    A = CT[26, :, :, :, :]
    A[np.where(A > 2000)] = np.random.randint(-150, -90)
    CT[26, :, :, :, :] = A
    f.close()
if __name__ != '__main__':
    trainPath = glob(os.path.join(trainDir, '*.h5'))[0]
    print(trainPath)
    f = h5py.File(trainPath, 'r')
    CT = f['img']
    mask = f['mask']
    mxv = -np.inf
    mnv = np.inf
    for i in range(len(filenames)):
        print(filenames[i])
        a = np.array(CT[i, :, :, :, :]).max()
        b = np.array(CT[i, :, :, :, :]).min()
        # if a > mxv:
        #     mxv = a
        print(a, b)
    # for i in range(len(filenames)):
    #     a = np.array(mask[i, :, :, :, :]).min()
    #     if a < mnv:
    #         mnv = a
    # print(mnv, mxv)

    # J = np.array(CT[26, :, :, :, :])
    # print(np.array(J).max())
    # # # # mask[np.where(mask != 1)] = 0
    # # # J[np.where(J > 3000)] = -950
    # # # print(J.max())
    # # for j in range(5000):
    # #     if J[j, :, :, :].max() == 3071:
    # #         break
    # # print(j)
    #
    #
    # # A = np.array(A)
    #
    # print(A.max())
    # # A = A.ravel()
    # # print(A.shape)
    # # plt.hist(A)
    # # plt.show()
    # # dset = f.create_dataset("img", data=A)
    #
    # f.close()
    # f = h5py.File(trainPath, 'w')
    #     f['img'] = CT
    # f.create_dataset('curA', data=A)
    #
if __name__ != '__main__':
    """test case to edit"""
    fpath = os.path.join(trainDir, 'SpleenDataTest.h5')
    ln = [[-2, 0, 8, 90],[20, 8, 4, -23]]
    with h5py.File(fpath, 'w') as f:
        f['test'] = ln
        # f['mask'] = trainMask
    f = h5py.File(fpath, 'r+')
    data = f['test']
    print(type(data))
    # data[np.where(data < 0)] = 100
    # f.close()
    # print(np.shape(data))
    V = data[1, :]
    V[np.where(V < 0)] = 100
    # print(V)
    data[1, :] = V
    f.close()
    f = h5py.File(fpath, 'r+')
    data = f['test']
    print(data[1, :])

if __name__ != '__main__':
    """taking 2D images"""
    ct = nib.load(os.path.join(trainImageDir, os.path.basename(trainImageDir) + filenames[1])).get_data()
    mask = nib.load(os.path.join(trainLabelDir, os.path.basename(trainLabelDir) + filenames[1])).get_data()
    print(filenames[3])
    mask[np.where(mask!=1)] = 0
    numPat = np.int(np.sum(mask)/100)
    # print(get_paired_patch2D(mask))
    # foreground = np.array(np.where(mask == 1))
    # centers = foreground[:, np.random.permutation(foreground.shape[1])[:5000]]
    # print(centers[2, :])
    # plt.plot(centers[2, :])
    # plt.show()
    ct_size = (128, 128, 3)
    mask_size = (128, 128, 3)
    trainCT, trainMask = get_paired_patch2D(ct, ct_size, mask, mask_size, numPat)
    print(np.shape(trainCT))
    print(np.shape(trainMask))
    trainMask = np.array(trainMask)
    demoCT = trainMask[405, :, :, 2]
    plt.imshow(demoCT)
    plt.show()

if __name__ != '__main__':
    # fInd = filenames[23]
    # print(filenames[fInd])
    trainCT = []
    trainMask = []
    # trainCT = np.empty_like()
    # trainMask  = np.empty_like()
    # ind = 0
    for fInd in filenames:
        print(fInd)
        ct = nib.load(os.path.join(trainImageDir, os.path.basename(trainImageDir) + fInd)).get_data()
        #     print(ct.shape)
        mask = nib.load(os.path.join(trainLabelDir, os.path.basename(trainLabelDir) + fInd)).get_data()
        mask[np.where(mask != 1)] = 0
        # numPat = np.int(np.sum(mask)/100)
        # print(numPat)
        # print(mask.shape, "mask")
        #     # plt.imshow(mask[:,:,-1], cmap='gray')
        #     # plt.show()
        #     # print(mask.shape)
        ct_size = (128, 128, 3)
        mask_size = (128, 128, 3)
        ctPatch, maskPatch = get_paired_patch2D(ct, ct_size, mask, mask_size, 1000)
        # print(np.shape(ctPatch))
        # print(np.shape(maskPatch))
        # if (np.shape(ctPatch) != ct_size) or (np.shape(maskPatch) != mask_size):
        #     print(np.shape(ctPatch))
        #     print(fInd)
        #     break
        # ctPatch = np.array(ctPatch).squeeze(0)
        # maskPatch = np.array(maskPatch).squeeze(0)

        trainCT.append(ctPatch)
        trainMask.append(maskPatch)
        # print("error free!")
        print("Train input shape:", np.shape(trainCT))
        print("Train mask shape:", np.shape(trainMask))
    #     # ind += 1
    #     # if ind == 2:
    #     #     break
    hpath = os.path.join(trainDir, 'SpleenData2D.h5')
    with h5py.File(hpath, 'w') as f:
        f['img'] = trainCT
        f['mask'] = trainMask
    # print(mrPatch[0].shape, maskPatch[0].shape)
    # mrPatch = np.array(mrPatch); maskPatch = np.array(maskPatch)
    # fig, ax = plt.subplots(1, 2, figsize=(64, 20))
    # ax[0].imshow(mrPatch[258, :, :, 2], cmap='gray')
    # ax[1].imshow(maskPatch[258, :, :, 2], cmap='gray')
    # plt.show()

    # print(get_paired_patch.__annotations__)

if __name__ != '__main__':
    totvox = []
    totspl = []
    splrat = []
    ind = 0
    for fInd in filenames:
        print(fInd)
        mask = nib.load(os.path.join(trainLabelDir, os.path.basename(trainLabelDir) + fInd)).get_data()
        mask[np.where(mask != 1)] = 0
        # print(np.sum(mask))
        # print(np.size(mask))
        totvox.append(np.size(mask))
        totspl.append(np.sum(mask))
        splrat.append((np.sum(mask)/np.size(mask)*100))
        # ind += 1
        # if ind == 2:
        #     break

    fig, ax = plt.subplots(2,1, figsize=(20, 10))
    ax[0].plot(totvox, 'b', linewidth=1, label='#total voxels')
    ax[0].plot(totspl, 'r', linewidth=1, label='#spleen voxels')
    ax[1].plot(splrat, 'g', linewidth=1, label='Ratio of spleen')
    ax[0].grid()
    ax[1].grid()
    leg0 = ax[0].legend(fontsize=20)
    leg1 = ax[1].legend(fontsize=20)
    plt.show()




