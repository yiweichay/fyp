import h5py
import numpy as np
from scipy.interpolate import RegularGridInterpolator, interp1d

class Temperature:
    """
    Calculates temperature based on intensity ratio and build plate position.

    Methods:
    - fit:
        x_mu (float): mean x position
        y_mu (float): mean y position

    - predict:
        R (np.array): ratio computed from image, passed as 1d vector

    Example useage:
        R = np.linspace(1, 3, 200)
        x_mu = 0
        y_mu = 0
        temp = Temperature(root='./config_matfile.mat')
        temp.fit(x_mu, y_mu)
        T_calculated = temp.predict(R)
    """

    def __init__(self, root):
        with h5py.File(root, 'r') as h:
            d_ratio = dict((k, item[:]) for (k, item) in h['ratiodata'].items())
        print(d_ratio)
        x_values = np.linspace(-100, 100, 5)
        y_values = np.linspace(-100, 100, 5)
        self.T_ref = d_ratio['T'].squeeze()
        self.n_Ts = len(self.T_ref)
        self.interp_R = RegularGridInterpolator((d_ratio['T'].squeeze(), x_values, y_values), 
                                            np.flip(d_ratio['R'], axis=2),
                                            bounds_error=False, 
                                            fill_value=np.nan, 
                                            method='linear')

    def fit(self, x_mu, y_mu):
        R_trans = self.interp_R(np.array([self.T_ref, [x_mu] * self.n_Ts, [y_mu] * self.n_Ts]).T)
        self.T_trans = interp1d(R_trans, self.T_ref, bounds_error=False, fill_value=np.nan)

    def predict(self, R):
        return self.T_trans(R)

'''
if __name__ == '__main__':

    import matplotlib.pyplot as plt
    #load the intensity ratio and plot
    with h5py.File('c:/Users/cyiwe/OneDrive - Imperial College London/ME4/FYP/fyp/Intensity Ratio1.mat', 'r') as f:
        print(f.keys())
        R = f['intensity_ratio']
        R = np.transpose(R, (2, 1, 0))
        print(R.shape)
        R = np.array(R).reshape(61440,)

    root = 'C:/Users/cyiwe/OneDrive - Imperial College London/ME4/FYP/fyp/config_matfile.mat'
    #R = np.linspace(0, 3, 200)
    x_mu = 0
    y_mu = 0
    T_model = Temperature(root=root)
    T_model.fit(x_mu, y_mu)
    T_calculated = T_model.predict(R)

    f, ax = plt.subplots(figsize=(5,5))
    #ax.plot(R, T_calculated, 'k')
    ax.scatter(R, T_calculated)
    ax.set_xlabel('Intensity ratio $I_1$ / $I_2$')
    ax.set_ylabel('Temperature (K)')
    #ax.set(xlim=(R.min(), R.max()), ylim=(0, 5000))
    ax.set(xlim=(0, 3), ylim=(0, 5000))
    ax.grid()
    plt.tight_layout()
    plt.show()
'''