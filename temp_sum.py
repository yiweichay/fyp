'''
This code is only used to plot the temperature summation graph for each layer, 
can also be used for other summation plots
'''

import pandas as pd
import h5py
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('classic')
from temp_intensity import Temperature

#Load and read files
df = pd.read_csv('C:/Users/cyiwe/OneDrive - Imperial College London/ME4/FYP/fyp/layer8/processed.csv')
x_pos = np.array(df['x_pos'][:])
y_pos = np.array(df['y_pos'][:])
t = np.array(df['time'][:])

#Read temperature array .mat file
temp = 'C:/Users/cyiwe/OneDrive - Imperial College London/ME4/FYP/fyp/Temperature Array_rotated.mat'

#Mark the center of the image
center_x = 160
center_y = 96

globMat = np.zeros((600,900))
pix = 0.02 #1 pixel = 0.02mm

def crop(img):
    img[:91,:] = 0
    img[101:,:] = 0
    img[:,:155] = 0
    img[:,165:] = 0
    return img

def mmToPixel(x, y):
    x = int(np.ceil((x-93)/pix) - 1)
    y = int(np.ceil(np.abs((y-(-71))/pix)) - 1)
    return x,y

with h5py.File(temp, 'r') as h:
    for idx in list(h.keys())[0:36500]:  
        T_calculated = np.fliplr(np.rot90(h[idx][:], 3)) #rotate clockwise by 90 degrees
        # T_calculated = crop(T_calculated)
        x_mu = x_pos[int(idx)]
        y_mu = y_pos[int(idx)] 
        globx, globy = mmToPixel(x_mu, y_mu)

        globMat_copy = np.zeros((600,900))
        if globMat_copy[globy-center_y:globy+center_y, globx-center_x:globx+center_x].shape != (192, 320):
            continue
        else:
            globMat_copy[globy-center_y:globy+center_y, globx-center_x:globx+center_x] = T_calculated
        globMat = np.add(globMat, globMat_copy)

        if int(idx) % 1000 == 0:
            print(int(idx))

#Print the temperature plot on build plate
f, ax = plt.subplots()
f.patch.set_facecolor('white')
# globMat[globMat == 0] = np.nan
globMat = globMat.tolist()
gm = ax.imshow(globMat)
cbar = plt.colorbar(gm)
ax.set_title('Temperature Summation without Rim: Layer 8')
cbar.ax.set_ylabel('Temperature (K)')
#plt.imsave('tempsum.png', globMat)
plt.show()
