'''
This code is to save the temperature arrays of each frame into a dictionary, saved as a .mat file
'''

import pandas as pd
import h5py
import numpy as np
import hdf5storage
import matplotlib.pyplot as plt
plt.style.use('classic')
from temp_intensity import Temperature
from scipy import ndimage

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
            if M[i,j] <= 500:
                c1[i,j] = 0
                c2[i,j] = 0
    return c1, c2

def crop(img):
    img[:91,:] = 0
    img[101:,:] = 0
    img[:,:155] = 0
    img[:,165:] = 0
    return img

def rotate(img, degree):
    img = ndimage.rotate(img, degree, reshape=False)
    return img

pth = 'C:/Users/cyiwe/OneDrive - Imperial College London/ME4/FYP/fyp/layer8/rawdataALLFRAMES.mat'

#Align images, calculate intensity ratio and save to .mat file
#Mark the center of the image
center_x = 160
center_y = 96

#Align images, calculate intensity ratio and save to .mat file
temp = {}
# cam1 = {}
# cam2 = {}

# globMat = np.zeros((600,800))
# pix = 0.02 #1 pixel = 0.02mm

for idx in range(25000, len(x_pos)):
    with h5py.File(pth, 'r') as h:
        #print(h.keys())
        c1 = h['cam1'][idx][:] #camera 1
        c2 = np.fliplr(h['cam2'][idx][:]) #camera 2
    
    #Check if the image pixels are correct
    if c1.shape and c2.shape != (192, 320):
        temp[str(idx).zfill(5)] = np.zeros((192,320))
        # cam1[str(idx).zfill(5)] = np.zeros((192,320))
        # cam2[str(idx).zfill(5)] = np.zeros((192,320))

    #Check if there are values for the laser position
    else:
        if np.all(centroid[idx]) == False:
            centroid[idx,0] = 144
            centroid[idx,1] = 109
            centroid[idx,2] = 130
            centroid[idx,3] = 89

        # temp[str(idx).zfill(5)] = np.zeros((192,320))
        # cam1[str(idx).zfill(5)] = np.zeros((192,320))
        # cam2[str(idx).zfill(5)] = np.zeros((192,320))
   
        delta_x1, delta_y1, delta_x2, delta_y2 = delta(center_x, center_y, idx)
        img_new1 = transform(delta_x1, delta_y1, c1)
        img_new2 = transform(delta_x2, delta_y2, c2)

        # Rotate the image by -110 degrees
        img_new1 = rotate(img_new1, -110)
        img_new2 = rotate(img_new2, -110)

        #Crop images to remain only the center of the melt pool
        # img_new1 = crop(img_new1)
        # img_new2 = crop(img_new2)

        #Save aligned images into .mat files
        # cam1[str(idx).zfill(5)] = img_new1
        # cam2[str(idx).zfill(5)] = img_new2
        
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
        temp[str(idx).zfill(5)] = T_calculated
        
        '''
        #Convert xpos and ypos to pixels
        globx = int(np.ceil((x_mu-93)/pix) - 1)
        globy = int(np.ceil(np.abs((y_mu-(-71))/pix)) - 1)

        #print('globx:', globx)
        #print('globy:', globy)
        globMat_copy = np.zeros((600,800))
        if globMat_copy[globy-center_y:globy+center_y, globx-center_x:globx+center_x].shape == (192, 320):
            globMat_copy[globy-center_y:globy+center_y, globx-center_x:globx+center_x] = T_calculated

        #globTemp[str(idx).zfill(5)] = globMat_copy
        globMat = np.add(globMat, globMat_copy)
        '''
        if idx % 1000 == 0:
            print(idx)

#Save the matlab file as hdf5 format
hdf5storage.savemat('Temperature Array_centroids_added', temp, format='7.3')
# hdf5storage.savemat('Camera1_alignedcrop_200mu', cam1, format='7.3')
# hdf5storage.savemat('Camera2_alignedcrop_200mu', cam2, format='7.3')
# hdf5storage.savemat('Camera1_rot-110', cam1, format='7.3')
# hdf5storage.savemat('Camera2_rot-110', cam2, format='7.3')
# hdf5storage.savemat('Temperature Array_rotated', temp, format='7.3')



'''
#Print the temperature plot on build plate
f, ax = plt.subplots()
#globMat = globMat.tolist()
gm = ax.imshow(globMat)
plt.imsave('test.png', globMat)
plt.colorbar(gm)
plt.show()
'''