import argparse
from os.path import join
from os.path import splitext
import sys
import numpy.lib.index_tricks as itricks
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from astropy.io import fits
import numpy as np
import treecorr
from scipy import special as spspec
from astropy.stats import bootstrap
from Wavefront import Image, WaveFitter, Telescope, TELESCOPE_DICT
from Wavefront import waveplot
from Wavefront import wavecorr
from Wavefront import utils
from sklearn.neighbors import KDTree

def log_likelihood(r, D, sigma, C, beta):
    constant = np.sum(np.log(1/(np.sqrt(2*np.pi)*sigma)))
    
    return constant + np.sum(-1*(C*r**(beta-2)-D)**2/(2.*np.square(sigma)))

def log_prior(C, beta):
    
    ## Flat priors 
    C_max = 100.*(2*np.pi/0.5)**2.
    C_min = 0.
    beta_max = 4.
    beta_min = 2.
    if (beta<beta_max)*(beta>beta_min)*(C<C_max)*(C>C_min):
        logp = np.log(1.0/(beta_max-beta_min)) + np.log(1.0/(C_max-C_min))
    else:
        logp = -np.inf
    return logp
    
def log_posterior(r, D, sigma, C, beta):
    return log_likelihood(r, D, sigma, C, beta) + log_prior(C, beta)

def gamma_beta(beta):
    ## A constant that depends on beta

    return 2**(beta-1)*spspec.gamma((beta+2)/2.)**2*spspec.gamma((beta+4)/2.)/(spspec.gamma(beta/2.)*spspec.gamma(beta+1))

def linStruct(image, mask, nbins=10, start=0.5, stop=40.):
    
    ## Build list of possible index values
    nx, ny = image.data.shape
    x_array = np.linspace(0, nx-1, nx)
    y_array = np.linspace(0, ny-1, ny)
    data = np.array([[x,y] for x in x_array for y in y_array if mask[x,y]==False])
    
    ## Initialize KDTree 
    tree = KDTree(data)
    
    ## Construct radial bins
    rbins = np.linspace(start, stop, nbins)
    rbins = [(rbins[i], rbins[i+1]) for i in range(len(rbins)-1)]
    
    ## Construct arrays to hold results  
    D = np.zeros(len(rbins))
    varD = np.zeros(len(rbins))
    r = np.array([image.pix_scale*(rbin[1]+rbin[0])/2. for rbin in rbins])
    
    ## Loop through each bin
    for i, rbin in enumerate(rbins):
        
        D_samples = []
        
        ## Loop through each point in the data
        for j, point in enumerate(data):
            
            ## Find data points within a radial shell
            high = tree.query_radius(point, rbin[1])
            low = tree.query_radius(point, rbin[0])
            shell = np.setxor1d(high[0], low[0])
            
            ## For each point in shell, calculate D_sample
            D_samples.extend([(image.data[tuple(point)]-image.data[tuple(data[k])])**2 for k in shell])
            
        ## Calculate structure function for rbin and bootstrap resample
        D_samples = np.array(D_samples)
        D_bootstraps = bootstrap(D_samples, bootnum=1000, samples = 500, bootfunc=np.mean)
        D[i] = D_samples.mean()
        varD[i]=D_bootstraps.var()
        
    return r, D, varD

def MCMC(r, D, varD, nsteps=10000, warmup=1000, C=100, beta=11/3., C_step=1, beta_step=0.1):

    # Initialize parameters and step sizes
    C_curr,beta_curr = C, beta
    logpost_curr = log_posterior(r,D,np.sqrt(varD),C_curr, beta_curr)        

    ## Create parameter vectors
    C = np.zeros(nsteps-warmup)
    beta = np.zeros(nsteps-warmup)

    for i in range(nsteps):

        ## Take step and evaluate
        C_new = C_curr + np.random.normal() *C_step
        beta_new = beta_curr + np.random.normal() * beta_step
        logpost_new = log_posterior(r,D,np.sqrt(varD),C_new, beta_new)

        ## Check if accepting
        if (np.exp(logpost_new - logpost_curr) > np.random.uniform()):
            C_curr = C_new
            beta_curr = beta_new
            logpost_curr = logpost_new

        ## Let the chain warm-up
        if i+1 > warmup:
            C[i-warmup] = C_curr
            beta[i-warmup] = beta_curr
            
    return C, beta

def main(file_name, image_range):

    ## Set range of images to use
    start = int(image_range)*1000
    stop = start + 1000

    ## Construct directory paths and filepaths
    data_file = file_name
    telescope_name = 'GPI'
    main_directory = '/nfs/slac/g/ki/ki19/lsst/snyder18'

    path_dict = utils.filepath_library(data_file, main_directory, telescope_name)

    ## Make GPI telescope
    telescope = TELESCOPE_DICT[telescope_name]
    pupil_mask = telescope.get_pupil((48,48), pix_scale=7.77/43)

    ## Get Images
    image_list = Image.import_fits(path_dict['Data'])

    rho_0_mp = []
    rho_0_var = []
    C_mp = []
    C_var = []
    beta_mp = []
    beta_var = []

    ## For each image in the range, read
    for image in image_list[start:stop]:

        ## Remove piston and convert to radians
        image.mean_center(mask=pupil_mask)
        image.data = image.data*2*np.pi/0.5

        r, D, varD = linStruct(image, pupil_mask, nbins=20, stop=13)
        C_samples, beta_samples = MCMC(r, D, varD, C_step=1., beta_step=0.05)

        rho_0_samples = (gamma_beta(beta_samples)/C_samples)**(1/(beta_samples-2.))

        rho_0_mp.append(rho_0_samples.mean())
        rho_0_var.append(rho_0_samples.var())
        C_mp.append(C_samples.mean())
        C_var.append(C_samples.var())
        beta_mp.append(beta_samples.mean())
        beta_var.append(beta_samples.var())

    rho_0_mp = np.array(rho_0_mp)
    rho_0_var = np.array(rho_0_var)
    C_mp = np.array(C_mp)
    C_var = np.array(C_var)
    beta_mp = np.array(beta_mp)
    beta_var = np.array(beta_var)

    save_file = join(path_dict['Results'], '{0}_MCMC_fit_{1}.csv'.format(path_dict['Base'], image_range))
    np.savetxt(save_file, np.transpose([C_mp, C_var, beta_mp, beta_var, rho_0_mp, rho_0_var]), delimiter=",")

if __name__ == '__main__':

    main(sys.argv[1], sys.argv[2])

