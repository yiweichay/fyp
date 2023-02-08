import pandas as pd
import h5py
import numpy as np
import hdf5storage
import matplotlib.pyplot as plt
#plt.style.use('classic')
from temp_intensity import Temperature

#Load and read files
df = pd.read_csv('C:/Users/cyiwe/OneDrive - Imperial College London/ME4/FYP/fyp/layer8/processed.csv')
x_pos = np.array(df['x_pos'][:])
y_pos = np.array(df['y_pos'][:])
t = np.array(df['time'][:])

#Read temperature array .mat file
temp = 'C:/Users/cyiwe/OneDrive - Imperial College London/ME4/FYP/fyp/Temperature Array.mat'

#Mark the center of the image
center_x = 160
center_y = 96

globMat = np.zeros((600,800))
pix = 0.02 #1 pixel = 0.02mm

#Input x and y position on build plate
#Loop through each frame
#Calculate global matrix
#Append temperature at specified pixel
#Need to associate with time too

def mmToPixel(x, y):
    x = int(np.ceil((x-93)/pix) - 1)
    y = int(np.ceil(np.abs((y-(-71))/pix)) - 1)
    return x,y

x = float(input("Input x position: "))
y = float(input("Input y position: "))
x,y = mmToPixel(x,y)
print(x,y)


temp_array = []
time = []

with h5py.File(temp, 'r') as h:
    for idx in list(h.keys())[0:2000]:  
        T_calculated = np.fliplr(np.rot90(h[idx][:], 3)) #rotate clockwise by 90 degrees
        x_mu = x_pos[int(idx)]
        y_mu = y_pos[int(idx)] 
        globx, globy = mmToPixel(x_mu, y_mu)
        
        globMat_copy = np.zeros((600,800))
        if globMat_copy[globy-center_y:globy+center_y, globx-center_x:globx+center_x].shape == (192, 320):
            globMat_copy[globy-center_y:globy+center_y, globx-center_x:globx+center_x] = T_calculated
        else:
            continue
        #globMat = np.add(globMat, globMat_copy)
        temp_array.append(globMat_copy[y,x])
        time.append(t[int(idx)])


#print(temp_array)
#print(time)

fig,ax = plt.subplots()
plt.plot(time, temp_array)
plt.show()
'''
#Print the temperature plot on build plate
f, ax = plt.subplots()
#globMat = globMat.tolist()
gm = ax.imshow(globMat)
plt.imsave('test.png', globMat)
plt.colorbar(gm)
plt.show()
'''