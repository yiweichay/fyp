import pandas as pd
import h5py
import numpy as np
import hdf5storage
import matplotlib.pyplot as plt
plt.style.use('classic')
from temp_intensity import Temperature

#Load and read files
df = pd.read_csv('C:/Users/cyiwe/OneDrive - Imperial College London/ME4/FYP/fyp/layer8/processed.csv')
x_pos = np.array(df['x_pos'][:])
y_pos = np.array(df['y_pos'][:])
t = np.array(df['time'][:])

#Read temperature array .mat file
temp = 'C:/Users/cyiwe/OneDrive - Imperial College London/ME4/FYP/fyp/Camera2_alignedcrop.mat'

#Mark the center of the image
center_x = 160
center_y = 96

globMat = np.zeros((600,900))
pix = 0.02 #1 pixel = 0.02mm

def mmToPixel(x, y):
    x = int(np.ceil((x-93)/pix) - 1)
    y = int(np.ceil(np.abs((y-(-71))/pix)) - 1)
    return x,y

with h5py.File(temp, 'r') as h:
    for idx in list(h.keys())[0:len(x_pos)]:  
        T_calculated = np.fliplr(np.rot90(h[idx][:], 3)) #rotate clockwise by 90 degrees
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
globMat = globMat.tolist()
gm = ax.imshow(globMat)
ax.set_title('Camera2 Cropped Integration Plot')
plt.colorbar(gm)
#plt.imsave('test.png', globMat)
plt.show()
