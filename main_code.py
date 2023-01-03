import pandas as pd
import h5py
import numpy as np
import matplotlib.pyplot as plt
from temp_intensity import Temperature

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

#Noise filtering function
def denoise(c1, c2, M):
    for i in range(0,192):
        for j in range(0,320):
            if M[i,j] <= 10000:
                c1[i,j] = 0
                c2[i,j] = 0
    return c1, c2

pth = 'C:/Users/cyiwe/OneDrive - Imperial College London/ME4/FYP/fyp/layer8/rawdataALLFRAMES.mat'

#Align images, calculate intensity ratio and save to .mat file
#Mark the center of the image
center_x = 160
center_y = 96

#Add a slider
from matplotlib.widgets import Slider

#Adjust the main plot to make room for the slider
f, ax = plt.subplots(2,3,figsize=(10,10))
f.tight_layout()
plt.subplots_adjust(bottom = 0.25)
ax_slider = f.add_axes([0.25, 0.1, 0.65, 0.03])
imgslider = Slider(ax=ax_slider, label='Frame Number', valmin=0, valmax=10000, valinit=0, valstep=1)
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
    img_new1crop = img_new1[70:130, 120:200] #crop image
    im1 = ax[0,0].imshow(img_new1)
    ax[0,0].grid(b=True, which='major', linestyle='-')
    ax[0,0].set_title(f"Laser Position: {np.round(centroid[idx,0],2)},{np.round(centroid[idx,1],2)}")

    img_new2crop = img_new2[70:130, 120:200] #crop image
    im2 = ax[1,0].imshow(img_new2)
    ax[1,0].grid(b=True, which='major', linestyle='-')
    ax[1,0].set_title(f"Laser Position: {np.round(centroid[idx,2],2)},{np.round(centroid[idx,3],2)}")

    
    #Check by multiplying the pixels
    M = np.multiply(img_new1, img_new2)
    print(M.max())
    print(np.where(M == M.max()))
    mul = ax[0,2].imshow(M)
    ax[0,2].grid(b=True, which='major', linestyle='-')
    ax[0,2].set_title('Multiplication of pixels c1xc2')

    #Calculating intensity ratios
    #Denoise the aligned images
    img_new1, img_new2 = denoise(img_new1, img_new2, M)

    R = np.divide(img_new2, img_new1)
    R[np.isnan(R)] = 0
    R[np.isinf(R)] = 0
    #print(np.where(R == R.max()))
    #print(R.max())
    new = R.copy()
    new[:70, :] = 0
    new[130:, :] = 0
    new[:, :120] = 0
    new[:, 200:] = 0
    new = new[70:130, 120:200] #crop image
    #print(new.tolist())
    #print(new.max())
    #print(np.where(new == new.max()))
    IR = ax[0,1].imshow(R, vmax=3)
    ax[0,1].grid(b=True, which='major', linestyle='-')
    ax[0,1].set_title('Intensity Ratio: c2/c1')

    #Plotting temperature - intensity graph
    #Reshape R into 1d vector
    R = np.reshape(R, (1, 61440))
    
    #include changing x and y position
    x_mu = x_pos[idx]
    y_mu = y_pos[idx]   
    #print(x_mu, y_mu)
    
    T_model = Temperature(root=root)
    T_model.fit(x_mu, y_mu)
    T_calculated = T_model.predict(R)
    ax[1,1].clear()
    ax[1,1].scatter(R, T_calculated)
    ax[1,1].set_xlabel('Intensity ratio $I_1$ / $I_2$')
    ax[1,1].set_ylabel('Temperature (K)')
    ax[1,1].set(xlim=(0, 3), ylim=(0, 5000))
    ax[1,1].grid(b=True, which='major', linestyle='-')

    #Setting colour bars
    global colorbar_set
    if not colorbar_set:
        plt.colorbar(im1)
        plt.colorbar(im2)
        plt.colorbar(IR)
        plt.colorbar(mul)
        colorbar_set = True

    #redraw the plot
    f.canvas.draw_idle() 

imgslider.on_changed(update)
plt.show()
