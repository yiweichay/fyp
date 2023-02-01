import pandas as pd
import h5py
import numpy as np
import hdf5storage
import matplotlib.pyplot as plt
plt.style.use('classic')
from temp_intensity import Temperature
from scipy.interpolate import RegularGridInterpolator, interp1d, LinearNDInterpolator

#Load and read files
df = pd.read_csv('C:/Users/cyiwe/OneDrive - Imperial College London/ME4/FYP/fyp/layer8/processed.csv')
x_pos = np.array(df['x_pos'][:])
y_pos = np.array(df['y_pos'][:])

#Temperature initialisation
root = 'C:/Users/cyiwe/OneDrive - Imperial College London/ME4/FYP/fyp/config_matfile.mat'

#Load centroids
pth = 'C:/Users/cyiwe/OneDrive - Imperial College London/ME4/FYP/fyp/layer8/weighted_centroids_diode_aligned.npy'
d = np.load(pth, allow_pickle=True).item()
centroid = d['c1_c2_centroid']
centroid = np.nan_to_num(centroid[:49400,:])

#Functions to align images
def delta(center_x, center_y, idx):
    delta_x1 = center_x - round(centroid[idx, 0])
    delta_y1 = center_y - round(centroid[idx, 1])
    delta_x2 = center_x - round(centroid[idx, 2])
    delta_y2 = center_y - round(centroid[idx, 3])
    return delta_x1, delta_y1, delta_x2, delta_y2

#Transformation function
def transform(delta_x, delta_y, img):
    #translate vertical
    if delta_y == 0:
        img_new = img.copy()
    else:
        temp = img[:-delta_y]
        img_new = img.copy()
        img_new[:delta_y] = img[-delta_y:]
        img_new[delta_y:] = temp
    
    #translate horizontal
    if delta_x == 0:
        img_new = img_new.copy()
    else:
        temp = img_new[:,-delta_x:]
        img_new[:,delta_x:] = img_new[:,:-delta_x]
        img_new[:,:delta_x] = temp
    
    return img_new

#Noise filtering function
def denoise(c1, c2, M):
    for i in range(0,192):
        for j in range(0,320):
            if M[i,j] <= 700:
                c1[i,j] = 0
                c2[i,j] = 0
    return c1, c2

pth = 'C:/Users/cyiwe/OneDrive - Imperial College London/ME4/FYP/fyp/layer8/rawdataALLFRAMES.mat'

#Align images, calculate intensity ratio and save to .mat file
#Mark the center of the image
center_x = 160
center_y = 96

#Align images, calculate intensity ratio and save to .mat file
temp_array = []

globMat = np.zeros((600,800))
pix = 0.02 #1 pixel = 0.02mm

for idx in range(0,len(x_pos)):
    with h5py.File(pth, 'r') as h:
        #print(h.keys())
        c1 = h['cam1'][idx][:] #camera 1
        c2 = np.fliplr(h['cam2'][idx][:]) #camera 2
    
    #Check if the image pixels are correct
    if c1.shape and c2.shape != (192, 320):
        continue
        #temp_array.append(np.zeros((1,)))
    
    #Check if there are values for the laser position
    elif np.all(centroid[idx]) == False:
        continue
        #temp_array.append(np.zeros((1,)))

    else:   
        delta_x1, delta_y1, delta_x2, delta_y2 = delta(center_x, center_y, idx)
        img_new1 = transform(delta_x1, delta_y1, c1)
        img_new2 = transform(delta_x2, delta_y2, c2)

        #Check by multiplying the pixels
        M = np.multiply(img_new1, img_new2)

        #Denoise the aligned images
        img_new1, img_new2 = denoise(img_new1, img_new2, M)
        #Calculating intensity ratios
        R = np.divide(img_new2, img_new1)
        R[np.isnan(R)] = 0
        R[np.isinf(R)] = 0

        R = np.reshape(R, [61440, 1])

        #include changing x and y position
        x_mu = x_pos[idx]
        y_mu = y_pos[idx]   

        T_model = Temperature(root=root)
        T_model.fit(x_mu, y_mu)
        T_calculated = T_model.predict(R)
        T_calculated = np.reshape(T_calculated, [192,320])
        T_calculated[np.isnan(T_calculated)] = 0
        T_calculated[np.isinf(T_calculated)] = 0

        #Save temperature to .mat file
        #temp_array.append(T_calculated)
    
        #Convert xpos and ypos to pixels
        globx = int(np.ceil((x_mu-93)/pix) - 1)
        globy = int(np.ceil(np.abs((y_mu-(-71))/pix)) - 1)

        #print('globx:', globx)
        #print('globy:', globy)
        globMat_copy = np.zeros((600,800))
        if globMat_copy[globy-center_y:globy+center_y, globx-center_x:globx+center_x].shape != (192, 320):
            continue
        else:
            globMat_copy[globy-center_y:globy+center_y, globx-center_x:globx+center_x] = T_calculated
        globMat = np.add(globMat, globMat_copy)

#temp = {"Temperature Array": np.array(temp_array)}

#Save the matlab file as hdf5 format
#hdf5storage.savemat('Temperature Array.mat', temp, format='7.3')

f, ax = plt.subplots()
#globMat = globMat.tolist()
gm = ax.imshow(globMat)
plt.colorbar(gm)
plt.show()