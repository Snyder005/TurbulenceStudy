import matplotlib
#matplotlib.use('Agg')

import scipy
import numpy as np
import sys
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import copy

def gsmooth(X, Y, var_Y=None, vexp=0.005, nsig = 3.0):

    ## Check Y variance
    if var_Y is None:
        var_Y = np.ones(len(Y))

    new_Y = np.zeros(len(X),float)

    ## Perform smoothing on each element of Y
    for i in range(len(X)):

        ## Construct Gaussian
        gaussian = np.zeros(len(X), float)
        sigma = vexp*X[i]

        ## Create window for smoothing
        sigrange = np.nonzero(abs(X-X[i]) <= nsig*sigma)
        gaussian[sigrange] = (1/(sigma*np.sqrt(2*np.pi)))*np.exp(-0.5*((X[sigrange]-X[i])/sigma)**2)

        ## Perform a weighted sum
        W_lambda = gaussian / var_Y
        W0 = np.sum(gaussian)
        W1 = np.sum(gaussian*Y)
        new_Y[i] = W1/W0

    return new_Y

def clip(X, Y, var_Y=None):

    clip_Y = copy.deepcopy(Y)

    var = np.ones(len(Y))

    sY = gsmooth(X, Y, var, vexp=0.01)
    err = abs(Y-sY)
    serr = gsmooth(X, err, var, 0.005, nsig=3.0)
    serr[serr == 0] = 1E-10

    index = np.where(err/serr > 6)
    bad_points = X[index]
 
    bad = np.array([],int)
    for i in range(len(bad_points)):
        bad = np.append(bad, np.where(abs(X - bad_points[i]) < 2))

    clip_Y[bad] = sY[bad]

    return clip_Y

def power_spec(Y, sample_spacing):

    n = len(Y)
    dT = sample_spacing

    P = np.fft.rfft(Y)
    norm = 2.0/n
    P = P*norm

    P2 = np.square(np.abs(P))
    K = np.fft.rfftfreq(n, d=dT)

    return K, P

def main(file_name):

    ## Read from file
    coefficients = np.loadtxt(file_name, dtype=np.float64, delimiter=',')

    ## Mean center zernike coefficients
    avg_coefficients = np.mean(coefficients, axis=0, keepdims=True)
    coefficients = coefficients - avg_coefficients

    N = coefficients.shape[0]
    dT = 0.001

    T = np.array([0.001*i for i in range(N)])

    P = np.zeros((coefficients.shape[1], N//2+1), dtype=np.complex128)
    W = []

    for i in range(coefficients.shape[1]):

        K, P[i,:] = power_spec(coefficients[:,i], dT)

    fig, axes = plt.subplots(nrows=3, ncols=3)
    fig.suptitle(r"Zernike Power Spectrums", fontsize=34)

    for i, ax in enumerate(axes.flatten()):

        ## This is range of known effect on focus
#        print K[1300:1400]
#        print P[i+1,1300:1400]

#        sP = gsmooth(K[1:], np.absolute(P[i+1,1:]), vexp=0.008, nsig=5)
#        new_logP = clip(K[1:], np.log(P[i+1, 1:]))

        ## Semilog plots of absolute real and imaginary parts
#        img = ax.semilogy(K[1:1500], np.absolute(P[i+1,1:1500].real), label="$|\mathrm{Re}(\mathcal{F})|$", alpha=0.5)
#        img = ax.semilogy(K[1:1500], np.absolute(P[i+1,1:1500].imag), 'r', label="$|\mathrm{Im}(\mathcal{F})|$", alpha=0.5)

        ## Plots of residual / smoothed residual
#        smooth_real = gsmooth(K[1:3000], P[i+1, 1:3000].imag)
#        absres = np.absolute(P[i+1, 1:3000].imag - smooth_real)
#        smooth_absres = gsmooth(K[1:3000], absres)
#        img = ax.plot(K[1:3000], absres/smooth_absres)

        ## Plots of clipped function
#        clip_P = clip(K[1:3000], P[i+1,1:3000].real)
#        img = ax.plot(K[1:3000], clip_P)

        ## Plot of |F(k)|^2 and smoothed function
        smoothed = gsmooth(K[1:], np.absolute(np.square(P[i+1,1:])))
        img = ax.loglog(K[1:], np.absolute(np.square(P[i+1, 1:])), K[1:], smoothed, 'r')

        ax.set_title(r"$Z_{{{}}}(t)$".format(i+2), fontsize=22)
        ax.minorticks_on()
        ax.grid(b=True, which='major', color='black', linestyle='-')
        ax.grid(b=True, which='minor', color='grey', linestyle='-')

    for ax in axes[2,:]:
        ax.set_xlabel(r'$f$ (Hz)', fontsize=22)
    for ax in axes[:,0]:
        ax.set_ylabel(r"Re($\mathcal{F}$) Res", fontsize=22)

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.show()

if __name__ == '__main__':

    main(sys.argv[1])
