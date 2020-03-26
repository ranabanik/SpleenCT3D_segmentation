import os
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import pickle

imgDir = r'/home/banikr2/PycharmProjects/SpleenCT3D_segmentation/Images'
dataDir = r'/home/banikr2/PycharmProjects/SpleenCT3D_segmentation/2020-03-25-01-19-13'
dataPath = glob(os.path.join(dataDir, '*.bin'))[0]
# print(dataPath)

with open(dataPath, 'rb') as pfile:
    h = pickle.load(pfile)
print(h.keys())

fig, ax = plt.subplots(figsize=(10, 10))

ax.plot(h['loss_train'], 'r', linewidth=1.5, label='Training loss')
ax.plot(h['loss_valid'], 'b', linewidth=1.5, label='Validation loss')
# ax.axis('equal')
ax.grid()
ax.set_xlabel('Epoch', fontsize=20)
ax.set_ylabel('Loss(a.u.)', fontsize=15)
leg = ax.legend(fontsize=20)

fig.savefig(os.path.join(imgDir, 'LossCurve2D46epochs.png'))
fig.show()

