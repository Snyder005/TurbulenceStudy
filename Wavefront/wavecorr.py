##
##  To Do List:
##
##  1. add multi-plot functionality to waveplot 
##  2. rewrite zernike_xi_t function
##  3. rewrite avg_xi_t function
##  4. rewrite using class objects
##


from wavefront import Image, WaveFitter
from telescope import Telescope, TELESCOPE_DICT
import utils
import copy
import waveplot
import treecorr
import numpy as np
from sklearn.neighbors import BallTree
import matplotlib.pyplot as plt

###############################################################################
##
## Correlation Functions - xi(r)
##
###############################################################################

def avg_xi_r(wavefront_list, telescope_name, pixscale=7.77/43, **kwargs):

    ## Get correlation parameters
    nbins = kwargs.pop('nbins', 15)
    min_sep = kwargs.pop('min_sep', 1.)
    max_sep = kwargs.pop('max_sep', 40)
    is_centered = kwargs.pop('is_centered', True)

    ## Deep copy to avoid altering reusable data
    wavefront_list = copy.deepcopy(wavefront_list)

    ## Check that data is a list
    if not isinstance (wavefront_list,list):
        wavefront_list = [wavefront_list]
    if len(wavefront_list) == 0:
        raise ValueError("No wavefronts provided")

    ## Get telescope from available telescopes in a dictionary
    shape = wavefront_list[0].data.shape
    telescope = TELESCOPE_DICT[telescope_name]
    pupil_mask = telescope.get_pupil(shape, pix_scale=pixscale)

    ## Make catalogs from wavefront list
    catalog_list = utils.image_to_catalog(wavefront_list, pupil_mask, 
                                          is_centered=is_centered)

    ## Calculate xi/varxi zero point (variance)
    xi_0 = np.zeros(len(wavefront_list))
    for i, wavefront in enumerate(wavefront_list):
        w = np.ma.MaskedArray(wavefront.data, mask=pupil_mask)
        xi_0[i] = np.var(w)
    varxi_0 = xi_0.var(dtype=np.float64)

    ## Initialize data vectors
    xi = np.zeros((nbins+1, len(wavefront_list)))
    varxi = np.zeros((nbins+1, len(wavefront_list)))
    r = np.zeros(nbins+1)

    xi[0,:] = xi_0
    r[0] = 0.0

    ## For each catalog calculate xi using treecorr
    for i, catalog in enumerate(catalog_list):
        dd = treecorr.KKCorrelation(nbins=nbins, min_sep=1., max_sep=40)
        dd.process(catalog)
        xi[1:,i] = dd.xi
        varxi[1:,i] = dd.varxi

    ## Find average xi/varxi
    avg_xi = np.mean(xi, axis=1)
    avg_varxi = np.mean(varxi, axis=1)
    avg_varxi[0] = varxi_0
    r[1:] = 7.77/43.*np.exp(dd.logr)

    ## Check save flags
    if 'save_file' in kwargs:
        np.savetxt(kwargs['save_file'], np.transpose([r, avg_xi, avg_varxi]))
    if 'save_graph' in kwargs:
        waveplot.line(r, avg_xi,**kwargs)

    return r, avg_xi, avg_varxi

def zernike_xi_r(coefficients, telescope_name, pixscale = 7.77/43*48/256., **kwargs):
    """Pixscale is smaller since zernikes are 256x256"""

    ## Read in file with zernike fit coefficients
    avg_coefficients = np.mean(coefficients, axis=0)

    ## Construct average zernike and run xi
    avg_zernike = WaveFitter.make_model(avg_coefficients, as_image=True)

    if 'save_image' in kwargs:
        telescope = TELESCOPE_DICT[telescope_name]
        pupil_mask = telescope.get_pupil((256, 256), pixscale)
        avg_zernike.display(mask=pupil_mask, title='Zernike Model for GPI Pupil',
                            cbar_label=r'$\mu \mathrm{{m}}$',
                            save_image=kwargs['save_image'],
                            display=False)

    r, xi, varxi = avg_xi_r(avg_zernike, telescope_name, pixscale, **kwargs)

    return r, xi, varxi
    
def residual_xi_r(wavefront_list, coefficients, telescope_name, pixscale = 7.77/43, **kwargs):

    ## Deep copy to avoid altering reusable data
    wavefront_list = copy.deepcopy(wavefront_list)
    
    ## Construct zernike model from average coefficients
    avg_coefficients = np.mean(coefficients, axis=0)

    avg_zernike = WaveFitter.make_model(avg_coefficients, as_image=True)
    avg_zernike.zoom(48/256.)

    ## Subtract average zernike from each wavefront
    for wavefront in wavefront_list:
        wavefront.data -= avg_zernike.data

    ## Calculate correlation function
    r, xi, varxi = avg_xi_r(wavefront_list, telescope_name, pixscale, **kwargs)

    return r, xi, varxi

def pupil_xi_r(telescope_name, **kwargs):

    rand_phase = Image(np.random.randn(256, 256))

   ## Get telescope from available telescopes in a dictionary
    telescope = TELESCOPE_DICT[telescope_name]
    pupil_mask = telescope.get_pupil((256, 256), pix_scale=7.77/43*48/256.)

    ## Make catalogs from wavefront list
    catalog = image_to_catalog(rand_phase, pupil_mask, is_centered=True)
    dd = treecorr.KKCorrelation(nbins=15, min_sep=1., max_sep=40)
    dd.process(catalog)

    xi = dd.xi
    varxi = dd.varxi
    r = 7.77/43.*np.exp(dd.logr)

    if 'save_file' in kwargs:
        np.savetxt(kwargs['save_file'], np.transpose([r, xi, varxi]))
    if 'save_graph' in kwargs:
        waveplot.line(r, xi, **kwargs)

    return r, x, varxi

###############################################################################
##
## Correlation Functions - xi(t)
##
###############################################################################

def avg_xi_t(wavefront_list, telescope_name, pixscale=7.77/43, **kwargs):

    ##### Add in calculation of zero point
   
    wavefront_list = copy.deepcopy(wavefront_list)

    ## Get telescope from available telescopes in a dictionary
    telescope = TELESCOPE_DICT[telescope_name]
    pupil_mask = telescope.get_pupil((48, 48), pix_scale=pixscale)

    ## Mean center all wavefronts
    for wavefront in wavefront_list:
        wavefront.mean_center(mask=pupil_mask)

    imagedata = np.array([wavefront.data for wavefront in wavefront_list])

    nT = imagedata.shape[0]
    nX = imagedata.shape[1]
    nY = imagedata.shape[2]

    x = np.zeros(nT)
    xi = []
    varxi = []

    for i in range(nX):
        for j in range(nY):
            if not pupil_mask[i, j]:
                t = []
                w = []
                for k in range(nT):
                    t.append(k)
                    w.append(imagedata[k, i, j])

                t = np.array(t)
                w = np.array(w)

                ## Create the data and random catalogs
                data = treecorr.Catalog(x=x, y=t, k=w)

                ## Create correlation objects
                dd = treecorr.KKCorrelation(nbins=50,  min_sep=1, max_sep=round(nT, -3))

                ## Process the catalogs
                dd.process(data)

                xi.append(dd.xi)
                varxi.append(dd.varxi)

    xi = np.array(xi)
    varxi = np.array(varxi)
    avg_xi = np.mean(xi, axis=0)
    avg_varxi = np.mean(varxi, axis=0)
    r = np.exp(dd.logr)

    if 'save_file' in kwargs:
        np.savetxt(kwargs['save_file'], np.transpose([r, avg_xi, avg_varxi]))
    if 'save_graph' in kwargs:
        waveplot.line(r, avg_xi, **kwargs) 

def zernike_xi_t(coefficients):

    ##### Add in calculation of zero point

    ## Mean center zernike coefficients
    avg_coefficients = np.mean(coefficients, axis=0, keepdims=True)
    coefficients = coefficients - avg_coefficients

    nT = coefficients.shape[0]
    nZ = coefficients.shape[1]

    xi = []

    x = np.zeros(nT)

    for i in range(nZ):

        t = [] 
        w = []

        for k in range(nT):
            t.append(k)
            w.append(coefficients[k,i])

        t = np.array(t)
        w = np.array(w)

        ## Create catalog and calculate correlation
        data = treecorr.Catalog(x=x, y=t, k=w)
        dd = treecorr.KKCorrelation(nbins=50,  min_sep=1, max_sep=round(nT, -3))
        dd.process(data)
        xi.append(dd.xi)

    logt = dd.logr

    fig1, axes1 = plt.subplots(nrows=2, ncols=3, sharex='col')
    fig1.suptitle(r"$\xi(\Delta t) = \langle a_i(t)a_i(t+\Delta t)\rangle $", fontsize=34)

    for i, ax in enumerate(axes1.flatten()):

        img = ax.plot(np.exp(logt), xi[i+1])
        ax.set_title(r"$Z_{{{}}}(t)$".format(i+2), fontsize=22)
        ax.minorticks_on()
        ax.grid(b=True, which='major', color='black', linestyle='-')
        ax.grid(b=True, which='minor', color='grey', linestyle='-')

    for ax in axes1[1,:]:
        ax.set_xlabel(r'$t$ (ms)', fontsize=22)
    for ax in axes1[:,0]:
        ax.set_ylabel(r"$\xi(\Delta t)$ ($\mu \mathrm{{m}}^2$)", fontsize=22)

    plt.subplots_adjust(top=0.85)

###############################################################################
##
## Structure Functions
##
###############################################################################

def corr2struct(r, xi, varxi, num_points=10, **kwargs):
    """Convert a correlation function to a structure function"""

    num = num_points

    struct = 2*(xi[0]-xi)
    ## Add modifications to varxi here
    varstruct = 2*(varxi[0]+varxi)

    C, success = leastsq(utils.residuals, [0, 1, 5/3.], args=(struct[0:num], r[0:num]))

    rfit = np.linspace(r[0], r[num]+1.0, 50)

    if 'save_file' in kwargs:
        np.savetxt(kwargs['save_file'], np.transpose([r, struct, varstruct]))

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.errorbar(r,struct, yerr=varstruct, label='Data')
    ax.plot(rfit, utils.powerlaw(C, rfit), color='red',
            label='{0:.4f}+{1:.4f}*r^{2:.4f}'.format(C[0], C[1], C[2]))
    ax.legend(loc='lower right')
    ax.minorticks_on()
    ax.grid(b=True, which='major', color='black', linestyle='-')
    ax.grid(b=True, which='minor', color='grey', linestyle='-')
    
    ax.set_title(r'Average Structure Function and Fit', y=1.04, fontsize=20)
    ax.set_xlabel(r'$\Delta r$ (m)', fontsize=18)
    ax.set_ylabel(r'$\mu \mathrm{{m}}^2$', fontsize=18)
    fig.tight_layout()

    if 'save_graph' in kwargs: plt.savefig(kwargs['save_graph'])

    return r, struct, varstruct

def struct_dt(T, Y, maxr, npoints=10.):

    bounds = np.linspace(0, maxr, npoints+1)
    bins = [(bounds[i], bounds[i+1]) for i in range(len(bounds)-1)]

    num_points = len(T)
    
    T = T[:,None]
    tree = BallTree(T)

    R = np.zeros(len(bins)+1)
    S_r = np.zeros(len(bins)+1)

    ## Iterate over each bin
    for i, bin in enumerate(bins):

        ## Get neighbors within upper and lower bound for each point in T
        rlow = tree.query_radius(T, r=bin[0])
        rhigh = tree.query_radius(T, r=bin[1])

        S = []

        ## Iterate over results for each point
        for j in range(num_points):

            ## Find neighbors within radial shell
            indices = np.setdiff1d(rhigh[j], rlow[j])
            if len(indices) > 0:
                S.append(np.mean(np.square(Y[indices]-Y[j])))

        R[i+1] = (bin[0]+bin[1])/2.
        S_r[i+1] = np.mean(S)

    plt.plot(R, S_r)
    plt.show()
                       
        

###############################################################################
##
## Debug Code
##
###############################################################################

if __name__ == '__main__':

    ## Debug code here
    T = np.linspace(0, 20, 200)
    Y = np.sin(T)
    print T
    print Y

    struct_dt(T, Y, 10, 20)
