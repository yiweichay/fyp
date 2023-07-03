import pandas as pd
import h5py
import numpy as np
import hdf5storage
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.stats as st
import statsmodels.api as sm
import csv


#Load and read files
df = pd.read_csv('C:/Users/cyiwe/OneDrive - Imperial College London/ME4/FYP/fyp/layer8/processed.csv')
x_pos = np.array(df['x_pos'][:])
y_pos = np.array(df['y_pos'][:])
power = np.array(df['diode'][:])
E_0 = np.array(df['E_0'][:])
t = np.array(df['time'][:])

#Read .mat file where cooling rates are saved
f = 'C:/Users/cyiwe/OneDrive - Imperial College London/ME4/FYP/fyp/Cooling Rates.mat'

cr = []
with h5py.File(f, 'r') as h:
    for i in list(h.keys())[0:len(x_pos)]:
        val = h[i][:]
        cr.append(val)
        
cr = np.array(cr)
cr = np.reshape(cr, [len(x_pos),])
# x = np.linspace(0,len(x_pos),len(x_pos))

'''
Plot Cooling Rate and Power
'''
fig, ax = plt.subplots(2,1)
ax[0].scatter(t[12700:12800],cr[12700:12800], s=5)
ax[0].axhline(y=0, color='r')
# ax[0].set_xlabel('Time (s)')
ax[0].set_ylabel('Temperature Gradient (K/$u$s)')

ax[1].plot(t[12700:12800], power[12700:12800])
ax[1].set_ylabel('Laser Power')
ax[1].set_xlabel('Time (s)')

# # Verify that it is not a normal distribution
# sm.qqplot(cr, line='45')

# ax.scatter(x_pos, y_pos, c=E_0, vmax=10, s=0.5)
# xpos, ypos = [], []
# def animate(i):
#     xpos.append(x_pos[i])
#     ypos.append(y_pos[i])
#     ax.scatter(xpos, ypos, color="red", s=0.5)

'''
Plot a histogram for cooling rates
'''
# plt.hist(cr[1:], bins=40, range=[-20,20])
# major_ticks = np.arange(-20, 20, 5)
# minor_ticks = np.arange(-20, 20, 1)
# ax.set_xticks(minor_ticks, minor=True)
# ax.set_yticks(minor_ticks, minor=True)
# ax.grid(which='both')
# ax.grid(which='minor', alpha=0.2)
# ax.grid(which='major', alpha=0.5)
# ax.set_xlabel('Cooling Rates')
# ax.set_ylabel('Frequency')
'''
Animate
'''
# fig, ax = plt.subplots()
# line, = ax.plot(x_pos, y_pos, color='k')

# def update(num, x, y, line):
#     line.set_data(x[:num], y[:num])
#     # line.axes.axis([0, 10, 0, 1])
#     # return line,

# ani = animation.FuncAnimation(fig, update, len(x_pos), fargs=[x_pos, y_pos, line],
#                               interval=0.5, repeat=False)
# # ani.save('build plate plot.gif')
# plt.show()

# f = r"C:/Users/cyiwe/OneDrive - Imperial College London/ME4/FYP/fyp/animation.mp4" 
# writervideo = animation.FFMpegWriter(fps=60) 
# ani.save(f, writer=writervideo)


'''
Plot build plate scatter plot
'''
# idx = 36350
# x_mu = x_pos[idx]
# y_mu = y_pos[idx]
# plt.scatter(x_pos[1:36450], y_pos[1:36450], s=0.1)
# plt.plot(x_mu, y_mu, marker='x', c='y')

# plt.plot(x_pos[150], y_pos[150], marker='x', c='r')
# plt.plot(x_pos[450], y_pos[450], marker='x', c='r')
# plt.plot(x_pos[790], y_pos[790], marker='x', c='r')
# plt.plot(x_pos[1140], y_pos[1140], marker='x', c='r')
# plt.plot(x_pos[1480], y_pos[1480], marker='x', c='r')
# plt.plot(x_pos[1870], y_pos[1870], marker='x', c='r')
# plt.plot(x_pos[2300], y_pos[2300], marker='x', c='r')
# plt.plot(x_pos[2830], y_pos[2830], marker='x', c='r')
# plt.plot(x_pos[3400], y_pos[3400], marker='x', c='r')
# plt.plot(x_pos[4050], y_pos[4050], marker='x', c='r')
# plt.plot(x_pos[4740], y_pos[4740], marker='x', c='r')
# plt.plot(x_pos[5460], y_pos[5460], marker='x', c='r')
# plt.plot(x_pos[6210], y_pos[6210], marker='x', c='r')
# plt.plot(x_pos[6970], y_pos[6970], marker='x', c='r')
# plt.plot(x_pos[7760], y_pos[7760], marker='x', c='r')
# plt.plot(x_pos[8550], y_pos[8550], marker='x', c='r')
# plt.plot(x_pos[9360], y_pos[9360], marker='x', c='r')
# plt.plot(x_pos[10190], y_pos[10190], marker='x', c='r')
# plt.plot(x_pos[11030], y_pos[11030], marker='x', c='r')
# plt.plot(x_pos[11880], y_pos[11880], marker='x', c='r')
# plt.plot(x_pos[12740], y_pos[12740], marker='x', c='r')
# plt.plot(x_pos[13610], y_pos[13610], marker='x', c='r')
# plt.plot(x_pos[14480], y_pos[14480], marker='x', c='r')
# plt.plot(x_pos[15360], y_pos[15360], marker='x', c='r')
# plt.plot(x_pos[16240], y_pos[16240], marker='x', c='r')
# plt.plot(x_pos[17120], y_pos[17120], marker='x', c='r')
# plt.plot(x_pos[18000], y_pos[18000], marker='x', c='r')
# plt.plot(x_pos[18890], y_pos[18890], marker='x', c='r')
# plt.plot(x_pos[19760], y_pos[19760], marker='x', c='r')
# plt.plot(x_pos[20650], y_pos[20650], marker='x', c='r')
# plt.plot(x_pos[21520], y_pos[21520], marker='x', c='r')
# plt.plot(x_pos[22390], y_pos[22390], marker='x', c='r')
# plt.plot(x_pos[23250], y_pos[23250], marker='x', c='r')
# plt.plot(x_pos[24100], y_pos[24100], marker='x', c='r')
# plt.plot(x_pos[24950], y_pos[24950], marker='x', c='r')
# plt.plot(x_pos[25770], y_pos[25770], marker='x', c='r')
# plt.plot(x_pos[26590], y_pos[26590], marker='x', c='r')
# plt.plot(x_pos[27390], y_pos[27390], marker='x', c='r')
# plt.plot(x_pos[28200], y_pos[28200], marker='x', c='r')
# plt.plot(x_pos[28970], y_pos[28970], marker='x', c='r')
# plt.plot(x_pos[29730], y_pos[29730], marker='x', c='r')
# plt.plot(x_pos[30460], y_pos[30460], marker='x', c='r')
# plt.plot(x_pos[31170], y_pos[31170], marker='x', c='r')
# plt.plot(x_pos[31870], y_pos[31870], marker='x', c='r')
# plt.plot(x_pos[32530], y_pos[32530], marker='x', c='r')
# plt.plot(x_pos[33150], y_pos[33150], marker='x', c='r')
# plt.plot(x_pos[33750], y_pos[33750], marker='x', c='r')
# plt.plot(x_pos[34310], y_pos[34310], marker='x', c='r')
# plt.plot(x_pos[34820], y_pos[34820], marker='x', c='r')
# plt.plot(x_pos[35300], y_pos[35300], marker='x', c='r')
# plt.plot(x_pos[35700], y_pos[35700], marker='x', c='r')
# plt.plot(x_pos[36070], y_pos[36070], marker='x', c='r')
# plt.plot(x_pos[36350], y_pos[36350], marker='x', c='r')

# curve = np.array([0, 150, 450, 790, 1140, 1480, 1870, 2300, 2830, 3400, 4050, 4740, 5460, 6210, 6970, 7760, 8550, 9360, 10190, 11030, 11880,
#          11030, 11880, 12740, 13610, 14480, 15360, 16240, 17120, 18000, 18890, 19760, 20650, 21520, 22390,
#          23250, 24100, 24950, 25770, 26590, 27390, 28200, 28970, 29730, 30460, 31170, 31870, 32530, 33150,
#          33750, 34310, 34820, 35300, 35700, 36070, 36350])

# print(curve.shape)

# res = np.ones((36450))
# r = 70 # Define the number of frames that i want to flag
# # Flag 1 when an edge is detected
# for i in curve:
#     res[i:i+r] = np.zeros((r))
# res = np.array(res)

#Save res in a new column in processed.csv
# data = pd.read_csv('C:/Users/cyiwe/OneDrive - Imperial College London/ME4/FYP/fyp/layer8/layer8_withoutRim.csv')
# data['edge'] = res.tolist()
# data.to_csv("./layer8/layer8_withoutRim.csv", index=False)

# plt.scatter(x_pos[1:36450], y_pos[1:36450], c=res[1:36450], s=1, cmap='viridis')


#Plot without the edges
edge = np.array(data['edge'][:])
diode = np.array(data['diode'][:])
x = []
y = []
cr_new = []
for i in cr:
    if i < -20 or i > 20:
        cr_new.append(0)
    else:
        cr_new.append(i)

crnew = []
for idx in range(0, 36450):
    if edge[idx] == 0 and diode[idx] >= 0.8: # and cr[idx] <= 0: #and cr[idx] >= -5:
        crnew.append(cr[idx])
        x.append(x_pos[idx])
        y.append(y_pos[idx])
    else:
        continue

# # Check for min and max cooling rates
print('Max cooling rate:', max(crnew[1:]))
print('Min cooling rate:', min(crnew[1:]))

# Smoothen the plot
# from scipy.signal import savgol_filter
# # crnew = savgol_filter(crnew, 25, 2, mode='nearest')
# plt.scatter(x[5:], y[5:], c=crnew[5:], s=15)

#plot where diode = 0 and where cr[idx] > 0
# xpower = []
# ypower = []
# for idx in range(0, 36450):
#     if diode[idx] == 0:
#         xpower.append(x_pos[idx])
#         ypower.append(y_pos[idx])
# plt.scatter(xpower, ypower, c='black', s=5, label='power = 0')

# cbar = plt.colorbar()
# cbar.ax.set_ylabel('Cooling Rates (K/mus)')
# plt.title('Plots for cooling rates, power below 0.8 removed')
# plt.legend()

plt.show()

