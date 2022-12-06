import pandas as pd
import h5py
import numpy as np
import matplotlib.pyplot as plt

import os
print(os.getcwd())

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

def transform(delta_x, delta_y, img):
    
    #translate vertical
    if delta_y == 0:
        img_new = img.copy()
    else:
        '''
        if delta_y > 0: #translate up
            delta_y = -delta_y
        '''
        temp = img[:-delta_y]
        img_new = img.copy()
        img_new[:delta_y] = img[-delta_y:]
        img_new[delta_y:] = temp
    
    #translate horizontal
    if delta_x == 0:
        img_new = img_new.copy()
    else:
        '''
        if delta_x > 0: 
            delta_x = -delta_x
            '''
        temp = img_new[:,-delta_x:]
        img_new[:,delta_x:] = img_new[:,:-delta_x]
        img_new[:,:delta_x] = temp
    
    return img_new

pth = 'C:/Users/cyiwe/OneDrive - Imperial College London/ME4/FYP/fyp/layer8/rawdataALLFRAMES.mat'

#Align images, calculate intensity ratio and save to .mat file
#Mark the center of the image
center_x = 160
center_y = 96
intensity_array = []

#Add a slider
from matplotlib.widgets import Slider

#Adjust the main plot to make room for the slider
f, ax = plt.subplots(2,2,figsize=(10,10))
plt.subplots_adjust(bottom = 0.25)

ax_slider = f.add_axes([0.25, 0.1, 0.65, 0.03])
imgslider = Slider(ax=ax_slider, label='Frame Number', valmin=0, valmax=1000, valinit=0, valstep=1)

def update(idx):
    with h5py.File(pth, 'r') as h:
        #print(h.keys())
        c1 = h['cam1'][idx][:] #camera 1
        c2 = np.fliplr(h['cam2'][idx][:]) #camera 2
    
     #Check if the image pixels are correct
    if c1.shape and c2.shape != (192, 320):
        return
    
    #Check if there are values for the laser position
    if np.all(centroid[idx]) == False:
        return
        
    delta_x1, delta_y1, delta_x2, delta_y2 = delta(center_x, center_y, idx)
    img_new1 = transform(delta_x1, delta_y1, c1)
    img_new2 = transform(delta_x2, delta_y2, c2)
    
    ax[0,0].imshow(img_new1)
    ax[1,0].imshow(img_new2)

    intensity = np.divide(img_new1, img_new2)
    intensity[np.isnan(intensity)] = 0
    intensity[np.isinf(intensity)] = 0
    ax[0,1].imshow(intensity, vmax=3)
    intensity_array.append(intensity)
    f.canvas.draw_idle() #redraw the plot
    
imgslider.on_changed(update)
plt.show()