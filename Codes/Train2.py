from PIL import Image
import h5py
import os
from glob import glob
import numpy as np
from random import gauss
from Utility import SpleenDataset, dice_loss, DiceLoss
import torch
import torch.nn as nn
from torch.utils import data
import torch.optim as optim
from torchvision.transforms import transforms
from model2D import UNet, weights_init
import time
import pickle
import matplotlib.pyplot as plt
import random

# torch.manual_seed(101)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

trainDir = r'/media/banikr2/DATA/SpleenCTSegmentation/Training'
projectDir = r'/home/banikr2/PycharmProjects/SpleenCT3D_segmentation'
modelDir = r'/media/banikr2/DATA/banikr_D_drive/model'
traindataPath = os.path.join(trainDir, 'Train512_27.h5')
validdataPath = os.path.join(trainDir, 'Valid512_03.h5')
# print(traindataPath, validdataPath)
F = h5py.File(traindataPath, 'r')
# print(F.keys()) #['img', 'mask']
trainCT = F['img']
trainMask = F['msk']
print(trainCT.shape, trainMask.shape)
G = h5py.File(validdataPath, 'r')
# print(G.keys())
validCT = G['img']
validMask = G['msk']
print(validCT.shape,validMask.shape) #(1936, 256, 256)

trainSet = SpleenDataset(trainCT, trainMask)
# img, mask = next(iter(trainSet))
# print(img.max(), img.min(), mask.max(), mask.min())
validSet = SpleenDataset(validCT, validMask)
# img, mask = next(iter(validSet))
# print(img.max(), img.min(), mask.max(), mask.min())
# print(np.array(trainCT).max())
# li = np.array([])
# for sl in range(np.array(trainMask).shape[0]):
#     li = np.append(li, np.sum(trainMask[sl, :, :]))
# plt.plot(li)
# plt.show()

tBatchSize = 8
vBatchSize = 8
max_epoch = 100
lr = 0.001
trainLoader = data.DataLoader(trainSet, batch_size=tBatchSize, shuffle=True, num_workers=4, drop_last=False)
validLoader = data.DataLoader(validSet, batch_size=vBatchSize, shuffle=False, num_workers=4, drop_last=True)
# print(len(validLoader))
# img, mask = next(iter(trainLoader))
# print(img.shape, mask.shape) #torch.Size([8, 1, 512, 512]) torch.Size([8, 1, 512, 512])
Net = UNet(in_dim=1, out_dim=1, num_filters=32).to(device)
Net.apply(weights_init)
Optimizer = optim.Adam(Net.parameters(), lr=lr)
# Criterion1 = nn.BCEWithLogitsLoss()# no cuda needed
# Criterion2 = dice_loss()    #DiceLoss()
# # Crit = nn.MSELoss()
#
# img, mask = next(iter(validLoader))
# img = img.float().to(device)
# mask = mask.float().to(device)
# out = Net(img)
# print(out.shape)
# Criterion2 = dice_loss()
# loss = Criterion1(out, mask) + Criterion2(out, mask)
# print(loss)
# epoch = 1
# printafterepoch = 10
# if (epoch % printafterepoch) == 1 and (epoch + 1) != 0:
#     mask = mask.detach().cpu().numpy().squeeze(1)
#     out = out.detach().cpu().numpy().squeeze(1)
#     for i in range(vBatchSize):  # validsize = 20
#         fig, ax = plt.subplots(1, 2, figsize=(8, 8))
#         # print("Happens")
#         ax[0].imshow(mask[i], cmap='gray')
#         ax[0].set_title("Mask")
#         ax[1].imshow(out[i], cmap='gray')
#         ax[1].set_title("Output")
#         savepath = os.path.join(projectDir, 'epoch{}_{}.png'.format(epoch, i))
#         fig.savefig(savepath)

if __name__ == '__main__':
    TIME_STAMP = time.strftime('%Y-%m-%d-%H-%M-%S')
    print(TIME_STAMP)
    FILEPATH_MODEL_LOAD = None
    """%%%%%%% Saving Criteria of Model %%%%%%%"""
    if FILEPATH_MODEL_LOAD is not None:  # Only needed for loading models or transfer learning
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
    # this_Project_log = os.path.join(dir_log, TIME_STAMP)
    # os.mkdir(this_Project_log)
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
            out = Net(ct)  # torch.Size([2, 512])
            # loss1 = Criterion1(out, mask)
            # loss = Criterion2(out, mask) #Criterion1(out, mask).cuda() +
            loss = dice_loss(out, mask)
            # loss = loss1 + loss2
            # print(loss)
            loss.backward()  # todo: loss.mean().backward for weighted loss
            Optimizer.step()  # here model weights get updated
            running_loss += loss.item()
            mean_loss = running_loss / (tbatch_idx + 1)
            time_compute = time.time() - time_compute_start
            time_batch = time_batch_load + time_compute
            running_time_batch += time_batch
            time_batch_avg = running_time_batch / (tbatch_idx + 1)
            print('epoch: {}/{}, batch: {}/{}, loss-train: {:.4f}, batch time taken: {:.2f} s, eta_epoch: {:.2f} hours'
                  .format(epoch + 1, max_epoch,
                          tbatch_idx + 1, len(trainLoader),
                          mean_loss,
                          time_batch,
                          time_batch_avg * (len(trainLoader) - (tbatch_idx + 1)) / 3600
                          )
                  )
            time_batch_start = time.time()
        loss_all_epoch_train.append(mean_loss)
        #
        # """%%%% If there is no validation loss the chosen criteria will be the lowest train loss
        #  and the model will have two states: train best and train latest %%%%"""
        running_loss = 0
        print('validating...')
        Net.eval()
        with torch.no_grad():
            for vbatch_idx, sample in enumerate(validLoader):
                ct = sample[0].float().to(device)
                mask = sample[1].float().to(device)
                out = Net(ct)
                # print(out.shape)
                # loss1 = Criterion1(out, mask)
                # loss = Criterion2(out, mask)
                # loss = loss1 + loss2
                loss = dice_loss(out, mask)
                running_loss += loss.item()
                mean_loss = running_loss / (vbatch_idx + 1)
                print('epoch: {}/{}, batch: {}/{}, mean-loss: {:.4f}'
                      .format(epoch + 1, max_epoch,
                              vbatch_idx + 1, len(validLoader),
                              mean_loss
                              )
                      )
        loss_all_epoch_valid.append(mean_loss)
        # print('loss epoch valid: {}'.format(loss_all_epoch_valid))
        printafterepoch = 10
        if (epoch % printafterepoch) == 1 and (epoch + 1) != 0:
            mask = mask.detach().cpu().numpy().squeeze(1)
            out = out.detach().cpu().numpy().squeeze(1)
            print(out.shape, mask.shape)
            for i in range(vBatchSize):  # validsize = 20
                fig, ax = plt.subplots(1, 2, figsize=(8, 8))
                # print("Happens")
                ax[0].imshow(mask[i, :, :], cmap='gray')
                ax[0].set_title("Mask")
                ax[1].imshow(out[i, :, :], cmap='gray')
                ax[1].set_title("Output")
                savepath = os.path.join(projectDir, 'epoch{}_{}.png'.format(epoch, i))
                fig.savefig(savepath)
            # break
        chosen_criteria = mean_loss
        print('Criteria at the end of epoch {} is {:.4f}'.format(epoch + 1, chosen_criteria))

        if chosen_criteria < model_save_criteria:
            print('criteria decreased from {:.4f} to {:.4f}, saving model ...'
                  .format(model_save_criteria, chosen_criteria))

            train_states_best = {
                'epoch': epoch + 1,
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
            'epoch': epoch + 1,
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
                   'Network architecture(layers used), Net = UNet(in_dims=1, out_dims=1,num_filters=64):\n{}'.format(
            [tBatchSize, vBatchSize],
            max_epoch, lr, k, 'dice_loss', Optimizer, Net))
