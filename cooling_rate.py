'''
Using the temperature arrays saved in the .mat file in 'temp_time.py',
This code calculates the instantaneous temperature derivatives (cooling rates) of all frames
This code can either:
1. Calculate the temperature derivative at a specific location and print the cooling curve
2. Calculate the temperature derivattive for all frames and save it in a .mat file
'''

import pandas as pd
import h5py
import numpy as np
import hdf5storage
import matplotlib.pyplot as plt
#plt.style.use('classic')

#Load and read files
df = pd.read_csv('C:/Users/cyiwe/OneDrive - Imperial College London/ME4/FYP/fyp/layer8/processed.csv')
x_pos = np.array(df['x_pos'][:])
y_pos = np.array(df['y_pos'][:])
t = np.array(df['time'][:])

#Read temperature array .mat file
temp = 'C:/Users/cyiwe/OneDrive - Imperial College London/ME4/FYP/fyp/Temperature Array_centroids_added.mat'

#Mark the center of the image
center_x = 160
center_y = 96

globMat = np.zeros((600,900))
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

# frame = int(input("Frame number:"))
# x = x_pos[frame]
# y = y_pos[frame]
# x,y = mmToPixel(x,y)
# x = float(input("Input x position: "))
# y = float(input("Input y position: "))

# print(x,y)


# temp_array = []
# time = []
cr = {}

for frame in range(0, len(x_pos)):
    x = x_pos[frame]
    y = y_pos[frame]
    x,y = mmToPixel(x,y)
    temp_array = []
    time = []
    with h5py.File(temp, 'r') as h:
        for idx in list(h.keys())[frame:frame+25]:  
            T_calculated = np.fliplr(np.rot90(h[idx][:], 3)) #rotate clockwise by 90 degrees
            x_mu = x_pos[int(idx)]
            y_mu = y_pos[int(idx)] 
            globx, globy = mmToPixel(x_mu, y_mu)

            globMat_copy = np.zeros((600,900))
            if globMat_copy[globy-center_y:globy+center_y, globx-center_x:globx+center_x].shape == (192, 320):
                globMat_copy[globy-center_y:globy+center_y, globx-center_x:globx+center_x] = T_calculated
            else:
                continue
            #globMat = np.add(globMat, globMat_copy)

            temp_array.append(globMat_copy[y,x])
            time.append(t[int(idx)])
        if int(frame) % 1000 == 0:
            print(int(frame))

    temp_array = np.array(temp_array)
    temp_array = temp_array.astype(float)
    temp_array[temp_array <= 30] = np.nan
    time = np.array(time)

    idx = np.isfinite(time) & np.isfinite(temp_array)
    if not idx.any():
        # print(frame, 'Cooling Rate:',0)
        cr[str(frame).zfill(5)] = 0

    else:
        slope, intercept = np.polyfit(time[idx], temp_array[idx], 1) #Line of best fit
        # print(frame, 'Cooling Rate:', slope*10**-6, 'Intercept:', intercept)
        cr[str(frame).zfill(5)] = slope*10**-6

hdf5storage.savemat('Temperature Gradient_centroids_added', cr, format='7.3') #roco is rate of change of lol

'''
To print the cooling rate graph for each frame
'''
# fig,ax = plt.subplots()
# slope, intercept = np.polyfit(time[idx], temp_array[idx], 1)
# print(frame, 'Cooling Rate:', slope*10**-6, 'Intercept:', intercept)
# plt.plot(time, temp_array, label='Cooling curve')
# plt.plot(time, slope*time+intercept, c='orange', label='Line of Best Fit')
# plt.legend()

# plt.xlabel('Time (s)')
# plt.ylabel('Temperature (K)')
# # plt.title('Cooling Curve: f79') 
# plt.show()
