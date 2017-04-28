## Author: Adam Snyder

##
##  Multiplot code currently under construction
##
##

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from scipy.signal import savgol_filter, periodogram

###############################################################################
##
##  Plot Functions
##
###############################################################################

def implot(image, display=False, **kwargs):
    """Plot an image with colorbar."""

    ## Create matplotlib figure 
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    cax = ax.imshow(image, origin='lower', interpolation='none')
    cbar = fig.colorbar(cax, orientation='vertical')

    ## Modify plot based on keyword arguments
    if 'title' in kwargs: ax.set_title(kwargs['title'], fontsize=24)
    if 'xlabel' in kwargs: ax.set_xlabel(kwargs['xlabel'], fontsize=16)
    if 'ylabel' in kwargs: ax.set_ylabel(kwargs['ylabel'], fontsize=16)
    if 'cbar_label' in kwargs: cbar.set_label(kwargs['cbar_label'], 
                                              fontsize=18)
    if 'save_image' in kwargs: plt.savefig(kwargs['save_image'])

    if display: plt.show()

def Zplot(Z, Z_noll, filebase):

    T = np.arange(0, Z.shape[0]/1000., 0.001)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    img = ax.plot(T, Z)
    ax.minorticks_on()
    ax.grid(b=True, which='major', color='black', linestyle='-')
    ax.set_ylabel(r'$a_{{{0}}}(t)$ $[\mu\mathrm{{m}}]$'.format(Z_noll), 
                  fontsize=24)
    ax.set_xlabel(r'$t$ $[\mathrm{s}]$', fontsize=24)
    ax.set_title(r'$Z_{{{0}}}$'.format(Z_noll), fontsize=30, y=1.04)
    
    filename = "{0}_Z{1}.png".format(filebase, Z_noll)
    plt.savefig(filename)
    plt.close(fig)

def PSDplot(periodogram, f, Z_noll, filebase):

    fig = plt.figure()
    ax = fig.add_subplot(111)
    smoothed = 10**savgol_filter(np.log10(periodogram), 101, 5)

## Perform linear fit to smoothed PSD
    par = np.polyfit(np.log10(f[f>1.]), np.log10(smoothed[f>1.]), 1)
    slope = par[0]
    intercept = par[1]

## Plot original PSD, smoothed PSD, and linear fit
    img = ax.loglog(f[1:], smoothed[1:], 'r',
                f[f>1.], (10**intercept)*f[f>1.]**slope, 'g')

    ax.legend(['Smoothed PSD', 'slope = {0:.2f}, intercept={1:.2f}'.format(slope, intercept)], loc=3, fontsize=18)

    ax.minorticks_on()
    ax.grid(b=True, which='major', color='black', linestyle='-')
    ax.set_ylabel(r'$\mathrm{{PSD}} \,\, [\mu\mathrm{{m}}^2/\mathrm{{Hz}}]$'.format(Z_noll), fontsize=24)
    ax.set_xlabel(r'$f$ $\mathrm{[Hz]}$', fontsize=24)
    ax.set_title(r'$Z_{{{0}}}$'.format(Z_noll), fontsize=30, y=1.04)

    filename = "{0}_PSD{1}.png".format(filebase, Z_noll)
    plt.savefig(filename)
    plt.close(fig)

    return slope, intercept
    
if __name__ == '__main__':

    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    y = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    line(x, y)
