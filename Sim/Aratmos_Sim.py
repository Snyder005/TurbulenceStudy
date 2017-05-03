
# coding: utf-8

# In[2]:

import numpy as np
import numpy.random as ra
import scipy.fftpack as sf
import pyfits as pf
import generate_grids as gg
import matplotlib.pyplot as plt
from datetime import datetime  


# In[3]:

exptime = 60.0 # Exposure time in seconds
rate = 1000.0 # Rate in Hz
alpha_mag = 0.999
n = 48
m = 1


# In[4]:

rootdir = './'

# filnename root for a multilayer simulation-worthy datacube
#    arfileroot = rootdir +'aratmos'+'_rate'+str(np.round(rate))+'_exptime'+str(exptime)+'_amag'+str(alpha_mag)
arfileroot = rootdir+"aratmos_rate{0}_exptime{1}_alpha{2}".format(np.round(rate), exptime, alpha_mag)
layerfileroot = arfileroot+'-layer'

## telescope geometry - Gemini for now. Move to structure for
bigD  = 8.4               ## primary diameter - 7.7 for Gemini, 8.4 for LSST
bigDs = 3                 ## inner M2 is 1.024 m

## derived quantities
bign      = n*m               ## width of phase screen for aperture

## for phase samples
pscale    = bigD/(n*m) ## pixel size (m) of samples in pupil plane
d         = pscale*m   ## subap diameter (m)

### make the aperture to impose later if desired
ax, ay    = gg.generate_grids(bign, scalefac=pscale)
ar        = np.sqrt(ax**2 + ay**2) ## aperture radius
ap_outer  = (ar <= bigD/2)
ap_inner  = (ar <= bigDs/2)   
aperture  = (ap_outer - ap_inner).astype(int)

timesteps = exptime * rate


# In[5]:

cp_params = np.array([
        #(0.40, 6.9, 284, 0),
        #              (0.78, 7.5, 267, 25),
        #              (1.07, 7.8, 244, 50),
        #              (1.12, 8.3, 267, 100),
        #              (0.84, 9.6, 237, 200),
        #              (0.68, 9.9, 232, 400),
        #              (0.66, 9.6, 286, 800),
        #              (0.91, 10.1, 293, 1600),
                      (0.40, 7.2, 270, 3400),
        #              (0.50, 16.5, 269, 6000),
        #              (0.85, 23.2, 59, 7600),
        #              (1.09, 32.7, 259, 13300),
        #              (1.08, 5.7, 320, 16000)
        ])

n_layers  = cp_params.shape[0]

r0s       = cp_params[:,0]              ## r0 in meters
vels      = cp_params[:,1]              ## m/s,  set to 0 to get pure boiling
dirs      = cp_params[:,2] * np.pi/180. ## in radians

## decompose velocities into components
vels_x    = vels * np.cos(dirs)
vels_y    = vels * np.sin(dirs)

## Higher velocities to test scaling
hvels_x = 2*vels * np.cos(dirs)
hvels_y = 2*vels * np.sin(dirs)


# In[ ]:

# generate spatial frequency grids
screensize_meters = bign * pscale
deltaf = 1./screensize_meters           ## spatial frequency delta
fx, fy = gg.generate_grids(bign, scalefac=deltaf, freqshift=True)

phase = np.zeros((bign,bign,n_layers,timesteps),dtype=float)
phFT  = np.zeros((bign,bign,n_layers,timesteps),dtype=complex)  ## array for FT of phase

phase_test = np.zeros((bign,bign,n_layers,timesteps),dtype=float)
phFT_test = np.zeros((bign,bign,n_layers,timesteps),dtype=complex)

for i in np.arange(n_layers):
    # Set the noise scaling powerlaw - the powerlaw below if from Johansson & Gavel 1994 for a 
    # Kolmogorov screen
    powerlaw = (2*np.pi*np.sqrt(0.00058)*(r0s[i]**(-5.0/6.0))*(fx**2. + fy**2.)**(-11.0/12.0)*bign*np.sqrt(np.sqrt(2.))/screensize_meters)
    powerlaw[0,0] = 0.0
    ## make array for the alpha parameter and populate it
    alpha_phase = - 2 * np.pi * (fx*vels_x[i] + fy*vels_y[i]) / rate

    alpha = alpha_mag * (np.cos(alpha_phase) + 1j * np.sin(alpha_phase))

    noisescalefac = np.sqrt(1 - (np.abs(alpha))**2)
    
    ## Testing changes in velocity
    halpha_phase = - 2 * np.pi * (fx*hvels_x[i] + fy*hvels_y[i]) / rate
    halpha = alpha_mag * (np.cos(halpha_phase) + 1j * np.sin(halpha_phase))
    hnoisescalefac = np.sqrt(1 - (np.abs(halpha))**2)
         
    for t in np.arange(timesteps):
        # generate noise to be added in, FT it and scale by powerlaw
        noise = np.random.randn(bign,bign)

        ## no added noise yet, start with a regular phase screen
        noiseFT = sf.fft2(noise) * powerlaw

        if t == 0:
            wfFT = noiseFT
            phFT[:,:,i,t] = noiseFT
            
            wfFT_test = noiseFT
            phFT_test[:,:,i,t] = noiseFT
        else:      
        # autoregression AR(1)
        # the new wavefront = alpha * wfnow + noise
            wfFT = alpha * phFT[:,:,i,t-1] + noiseFT * noisescalefac
            phFT[:,:,i,t] = wfFT
            
            wfFT_test = halpha * phFT_test[:,:,i,t-1] + noiseFT * hnoisescalefac
            phFT_test[:,:,i,t] = wfFT_test          
            
        # the new phase is the real_part of the inverse FT of the above
        wf = sf.ifft2(wfFT).real
        phase[:,:,i,t] = wf

        # Test for velocity change
        wf_test = sf.ifft2(wfFT_test).real
        phase_test[:,:,i,t] = wf_test

phaseout = np.sum(phase, axis=2)  # sum along layer axis
hdu = pf.PrimaryHDU(phaseout.transpose())
hdu.writeto(arfileroot+'.fits', clobber=True)

phaseout_test = np.sum(phase_test, axis=2)
hdu_test = pf.PrimaryHDU(phaseout_test.transpose())
hdu_test.writeto(arfileroot+'test.fits', clobber=True)


# In[ ]:



