import numpy as np
import matplotlib.pyplot as plt
import os

raw_data = np.fromfile('./vtd_kitti_format/training/velodyne/100.bin',dtype=np.float32)
raw_data = raw_data.reshape(-1,4)
print(raw_data)
plt.plot(raw_data[:,0],raw_data[:,1],'ro')
plt.show()

