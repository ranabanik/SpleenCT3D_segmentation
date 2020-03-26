from PIL import Image
import h5py
import os
from glob import glob
import numpy as np
from random import gauss
from Utility import SpleenDataset, dice_loss
import torch
import torch.nn as nn
from torch.utils import data
import torch.optim as optim
# from torchvision import transforms
from torchvision.transforms import transforms
from model2D import UNet, weights_init
import time
import pickle
import matplotlib.pyplot as plt
import random

torch.manual_seed(101)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

dataDir = r'/media/banikr2/DATA/SpleenCTSegmentation'
projectDir = r'/home/banikr2/PycharmProjects/SpleenCT3D_segmentation'
modelDir = r'/media/banikr2/DATA/banikr_D_drive/model'
traindataPath = os.path.join(dataDir, 'SpleenTilesNonOverLapTrain.h5')
validdataPath = os.path.join(dataDir, 'SpleenTilesNonOverLapValid.h5')
print(traindataPath, validdataPath)
F = h5py.File(traindataPath, 'r')
print(F.keys()) #['img', 'mask']
trainCT = F['img']
trainMask = F['msk']
print(trainCT.shape), print(trainMask.shape) #(13180, 256, 256)
G = h5py.File(validdataPath, 'r')
print(G.keys())
validCT = G['img']
validMask = G['msk']
print(validCT.shape), print(validMask.shape) #(1936, 256, 256)
# img = np.array(CT[0, 0, :, :, :])
# print(img.shape, img.dtype)
# # img = getSWN(img, 10, 10)
# print(type(img), img.dtype)
# # img = np.concatenate((img[:,:,np.newaxis], img[:, :, np.newaxis], img[:, :, np.newaxis]), axis=2)
# # img = np.transpose(img, (2, 0, 1))
# if __name__ != '__main__':
#     breakwhen = 0
#     # spleensum = np.array([])
#     for i in range(CT.shape[0]):
#         breakwhen += 1
#         # spleensum = np.append(spleensum, np.sum(Mask[i,:,:]))
#         if np.sum(Mask[i, :, :]) > 1000:
#             break
#     print(breakwhen)
#     img = Image.fromarray(CT[5070, :, :])
#     msk = Mask[5070, :, :]
#     msk = Image.fromarray(np.uint8(255*msk))#.convert('1')
#     # msk = ToTensor()(msk)
#     # print(msk.std())
#     # print(msk.max(), msk.min())
#     # # print(np.shape(img))
#     # # print(msk.max())
#
#     XForm = transforms.RandomHorizontalFlip()
#     imgxf = XForm(img)
#     img.show()
#     msk.show()
#     imgxf.show()
#     # mskxf.show()
#     plt.imshow(Mask[5070, :, :], cmap='gray')
#     plt.show()
#     trainCT = CT[0:15116-1132, :, :]
#     trainMask = Mask[0:15116-1132, :, :]
#     validCT = CT[15116-1132:, :, :]
#     validMask = Mask[15116-1132:, :, :]
#     print(trainCT.shape, trainMask.shape, validCT.shape, validMask.shape)

#     newTrainCT = []
#     newTrainMask = []
#     uselessCT = []
#     uselessMask = []
#
#     for i in range(trainCT.shape[0]):
#         if np.sum(trainMask[i, :, :]) > 0:
#             newTrainMask.append(trainMask[i, :, :])
#             newTrainCT.append(trainCT[i, :, :])
#         else:
#             uselessCT.append(trainCT[i, :, :])
#             uselessMask.append(trainMask[i, :, :])
#
#     newTrainCT = np.array(newTrainCT)
#     newTrainMask = np.array(newTrainMask)
#     print(newTrainMask.shape, newTrainCT.shape) #(1204, 256, 256)
#     print(np.shape(uselessCT), np.shape(uselessMask)) #(12780, 256, 256)
#
#     randomUseless = random.sample(range(0, 12780), 4000)
#     # print(randomUseless)
#
#     newUselessCT = []
#     newUselessMask = []
#     uselessCT = np.array(uselessCT)
#     uselessMask = np.array(uselessMask)
#     for x in randomUseless:
#         # j = uselessCT[x, :, :]
#         newUselessCT.append(uselessCT[x, :, :])
#         newUselessMask.append(uselessMask[x, :, :])
#     newUselessCT = np.array(newUselessCT)
#     newUselessMask = np.array(newUselessMask)
#     print(newUselessCT.shape, newUselessMask.shape)
#
#     trainCT = np.append(newTrainCT, newUselessCT, axis=0)#.unsqueeze(1)
#     trainMask = np.append(newTrainMask, newUselessMask, axis=0)#.unsqueeze(1)
#     print("finally: ", trainCT.shape, trainMask.shape)
#
#     xForm = transforms.Compose([transforms.RandomHorizontalFlip(),
#                               transforms.RandomVerticalFlip(),
#                               transforms.RandomAffine(20),
#                               transforms.ToTensor()])
#
#     trainSet = SpleenDataset(trainCT, trainMask, transform=xForm) #works for None type also #validation ok #test ok #train ok
#
#      #torch.Size([1, 128, 128]) torch.Size([1, 128, 128]) #torch.Size([1, 1, 256, 256])
#
#     validSet = SpleenDataset(validCT, validMask, transform=None)
#     img, mask = next(iter(validSet))
#
#     print(img.max(), mask.max())
#     print(img.shape, mask.shape)
tBatchSize = 20
vBatchSize = 20
max_epoch = 100
lr = 0.0001
trainSet = SpleenDataset(validCT, validMask)
img, mask = next(iter(validSet))
print(img.max(), img.min(), mask.max(), mask.min())
validSet = SpleenDataset(validCT, validMask)
trainLoader = data.DataLoader(trainSet, batch_size=tBatchSize, shuffle=True, num_workers=4)

img, mask = next(iter(trainSet))
print(img.max(), img.min(), mask.max(), mask.min())
validLoader = data.DataLoader(validSet, batch_size=vBatchSize, shuffle=False, num_workers=4)
if __name__ != '__main__':
    # trainCT = CT[0:27, :, :, :, :]
    # trainMask = Mask[0:27, :, :, :, :]
    # validCT = CT[27:, :, :, :, :]
    # validMask = Mask[27:, :, :, :, :]

    # trainCT = np.transpose(trainCT, (0, 1, 4, 2, 3))
    # trainCT = trainCT.reshape(-1, 128, 128)
    # trainMask = np.transpose(trainMask, (0, 1, 4, 2, 3))
    # trainMask = trainMask.reshape(-1, 128, 128)
    # validCT = np.transpose(validCT, (0, 1, 4, 2, 3))
    # validCT = validCT.reshape(-1, 128, 128)
    # validMask = np.transpose(validMask, (0, 1, 4, 2, 3))
    # validMask = validMask.reshape(-1, 128, 128)
    # print(trainCT.shape, trainMask.shape, validCT.shape, validMask.shape)
    #(81000, 128, 128) (81000, 128, 128) (9000, 128, 128) (9000, 128, 128)
    # img, mask = next(iter(trainSet))


    img, mask = next(iter(validLoader))
    print(img.shape, mask.shape) #torch.Size([20, 1, 128, 128]) torch.Size([20, 1, 128, 128])

    Net = UNet(in_dim=1, out_dim=1, num_filters=64).to(device)
    Net.apply(weights_init)
    Optimizer = optim.Adam(Net.parameters(), lr=lr)
# weight = torch.tensor(10.)
# Criterion = nn.BCEWithLogitsLoss(pos_weight=weight)
# Criterion = dice_loss()
# epoch = 11
#
# img = img.float().to(device)
# mask = mask.float().to(device)
# out = Net(img)
# print(out.shape, out.max(), out.min())

# Criterion = dice_loss(out, mask).cuda()
# print(Criterion)

    # img, mask = next(iter(trainLoader))
    # img = img.float().to(device)
    # mask = mask.float().to(device)
    # out = Net(img)
    # print(out)
    # loss = Criterion(out, mask)
    # print(loss)
if __name__ != '__main__':
    TIME_STAMP = time.strftime('%Y-%m-%d-%H-%M-%S')
    print(TIME_STAMP)
    FILEPATH_MODEL_LOAD = None
    """%%%%%%% Saving Criteria of Model %%%%%%%"""
    if FILEPATH_MODEL_LOAD is not None: #Only needed for loading models or transfer learning
        train_states = torch.load(FILEPATH_MODEL_LOAD)
        Net.load_state_dict(train_states['train_states_latest']['model_state_dict'])
        Optimizer.load_state_dict(train_states['train_states_latest']['optimizer_state_dict'])
        train_states_best = train_states['train_states_best']
        loss_valid_min = train_states_best['loss_valid_min']
        model_save_criteria = train_states_best['model_save_criteria']

    else:
        train_states = {}
        model_save_criteria = np.inf

    FILEPATH_MODEL_SAVE = os.path.join(modelDir, '{}.pt'.format(TIME_STAMP))
    this_project_dir = os.path.join(projectDir, TIME_STAMP)
    os.mkdir(this_project_dir)
    #this_Project_log = os.path.join(dir_log, TIME_STAMP)
    #os.mkdir(this_Project_log)
    FILEPATH_LOG = os.path.join(this_project_dir, '{}_LogMetric.bin'.format(TIME_STAMP))

    loss_all_epoch_train = []
    loss_all_epoch_valid = []
    """%%%%%%% Initiate training %%%%%%%"""
    for epoch in range(max_epoch):
        print("training...")
        running_loss = 0
        running_time_batch = 0
        time_batch_start = time.time()
        Net.train()
        for tbatch_idx, sample in enumerate(trainLoader):
            time_batch_load = time.time() - time_batch_start
            time_compute_start = time.time()
            ct = sample[0].float().to(device)
            mask = sample[1].float().to(device)
            Optimizer.zero_grad()
            out = Net(ct)  #torch.Size([2, 512])
            loss = dice_loss(out, mask).cuda()
            # print(loss)
            loss.backward() #todo: loss.mean().backward for weighted loss
            Optimizer.step() #here model weights get updated
            running_loss += loss.item()
            mean_loss = running_loss/(tbatch_idx+1)
            time_compute = time.time() - time_compute_start
            time_batch = time_batch_load + time_compute
            running_time_batch += time_batch
            time_batch_avg = running_time_batch/(tbatch_idx+1)
            print('epoch: {}/{}, batch: {}/{}, loss-train: {:.4f}, batch time taken: {:.2f} s, eta_epoch: {:.2f} hours'
                  .format(epoch+1, max_epoch,
                          tbatch_idx+1, len(trainLoader),
                          mean_loss,
                          time_batch,
                          time_batch_avg * (len(trainLoader) - (tbatch_idx+1))/3600
                          )
                  )
            time_batch_start = time.time()
        loss_all_epoch_train.append(mean_loss)

        """%%%% If there is no validation loss the chosen criteria will be the lowest train loss
         and the model will have two states: train best and train latest %%%%"""
        running_loss = 0
        print('validating...')
        Net.eval()
        with torch.no_grad():
            for vbatch_idx, sample in enumerate(validLoader):
                ct = sample[0].float().to(device)
                mask = sample[1].float().to(device)
                out = Net(ct)
                loss = dice_loss(out, mask)
                running_loss += loss.item()
                mean_loss = running_loss/(vbatch_idx+1)
                print('epoch: {}/{}, batch: {}/{}, mean-loss: {:.4f}'
                      .format(epoch+1, max_epoch,
                              vbatch_idx+1, len(validLoader),
                              mean_loss
                              )
                      )
        loss_all_epoch_valid.append(mean_loss)
        # print('loss epoch valid: {}'.format(loss_all_epoch_valid))
        printafterepoch = 10
        fig, ax = plt.subplots(1, 2, figsize=(8, 8))
        for i in range(vBatchSize):  # validsize = 20
            ax[0].imshow(mask[i].detach().cpu().numpy().squeeze(0), cmap='gray')
            ax[1].imshow(out[i].detach().cpu().numpy().squeeze(0))
            # ax.set_title("Valid MSE loss at {}: {}".format(epoch,
            #            dice_loss(mask[i].detach().cpu().numpy(),
            #                      out[i].detach().cpu().numpy())))
            # ax.legend()
            # ax.grid()
            # plt.show()
            savepath = os.path.join(projectDir, '{}.png'.format(epoch))
            fig.savefig(savepath)
            # break
        chosen_criteria = mean_loss
        print('Criteria at the end of epoch {} is {:.4f}'.format(epoch+1, chosen_criteria))

        if chosen_criteria < model_save_criteria:
            print('criteria decreased from {:.4f} to {:.4f}, saving model ...'
                  .format(model_save_criteria, chosen_criteria))

            train_states_best = {
                'epoch': epoch+1,
                'model_state_dict': Net.state_dict(),
                'optimizer_state_dict': Optimizer.state_dict(),
                'model_save_criteria': chosen_criteria
            }
            train_states['train_states_best'] = train_states_best
            torch.save(train_states, FILEPATH_MODEL_SAVE)
            model_save_criteria = chosen_criteria

        log = {
            'loss_train': loss_all_epoch_train,
            'loss_valid': loss_all_epoch_valid,
        }

        with open(FILEPATH_LOG, 'wb') as pfile:
            pickle.dump(log, pfile)

        train_states_latest = {
            'epoch': epoch+1,
            'model_state_dict': Net.state_dict(),
            'optimizer_state_dict': Optimizer.state_dict(),
            'model_save_criteria': chosen_criteria
        }
        train_states['train_states_latest'] = train_states_latest
        torch.save(train_states, FILEPATH_MODEL_SAVE)
    print(TIME_STAMP)
    k = 0
    FILEPATH_config = os.path.join(this_project_dir, 'config.txt')
    with open(FILEPATH_config, 'w') as file:
        file.write('Batch size:{}\n'
                   'Epochs:{}\n'
                   'Learning rate:{}\n'
                   'Cross validation #folds:{}\n'
                   'Criterion:{}\n'
                   'Optimizer:{}\n'
                   'Network architecture(layers used), Net = UNet(in_dims=1, out_dims=1,num_filters=64):\n{}'.format([tBatchSize, vBatchSize],
                    max_epoch, lr, k, 'dice_loss', Optimizer, Net))