import pandas as pd
import h5py
import numpy as np
import matplotlib.pyplot as plt
from temp_intensity import Temperature

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
#intensity_array = []

#Temperature initialisation
root = 'C:/Users/cyiwe/OneDrive - Imperial College London/ME4/FYP/fyp/config_matfile.mat'
x_mu = 0
y_mu = 0
T_model = Temperature(root=root)
T_model.fit(x_mu, y_mu)

#Add a slider
from matplotlib.widgets import Slider

#Adjust the main plot to make room for the slider
f, ax = plt.subplots(2,2,figsize=(10,10))
plt.subplots_adjust(bottom = 0.25)
ax_slider = f.add_axes([0.25, 0.1, 0.65, 0.03])
imgslider = Slider(ax=ax_slider, label='Frame Number', valmin=0, valmax=1000, valinit=0, valstep=1)
global colorbar_set #this is like volatile in C, for interrupts
colorbar_set = False

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
    
    #Image alignment
    delta_x1, delta_y1, delta_x2, delta_y2 = delta(center_x, center_y, idx)
    img_new1 = transform(delta_x1, delta_y1, c1)
    img_new2 = transform(delta_x2, delta_y2, c2)
    im1 = ax[0,0].imshow(img_new1)
    im2 = ax[1,0].imshow(img_new2)

    #Calculating intensity ratios
    R = np.divide(img_new1, img_new2)
    R[np.isnan(R)] = 0
    R[np.isinf(R)] = 0
    IR = ax[0,1].imshow(R, vmax=3)
    #R = np.array(R).reshape(61440,0)

    #Plotting temperature - intensity graph
    T_calculated = T_model.predict(R)
    ax[1,1].scatter(R, T_calculated)
    ax[1,1].set_xlabel('Intensity ratio $I_1$ / $I_2$')
    ax[1,1].set_ylabel('Temperature (K)')
    ax[1,1].set(xlim=(0, 3), ylim=(0, 5000))
    ax[1,1].grid()
    #plt.tight_layout()

    #Setting colour bars
    global colorbar_set
    if not colorbar_set:
        plt.colorbar(im1)
        plt.colorbar(im2)
        plt.colorbar(IR)
        colorbar_set = True

    #intensity_array.append(R)
    #redraw the plot
    f.canvas.draw_idle() 
    
imgslider.on_changed(update)
plt.show()
