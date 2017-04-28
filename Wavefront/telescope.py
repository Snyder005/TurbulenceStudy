## 
##  Telescope object goes here
##
##
##
##

import numpy as np
import sys

###############################################################################
##
## Telescope Class
##
###############################################################################

class Telescope():
    """Class describing the physical dimensions of a telescope pupil."""

    def __init__(self, name, outD, inD):

        self.name = name
        self.outD = outD
        self.inD = inD

    def __repr__(self):
        return 'Telescope({0},{1},{2})'.format(self.name, self.outD, self.inD)

    def __str__(self):
        return 'Telescope: {0}'.format(self.name)

    def get_pupil(self, imShape, pixscale, whole=False, freqshift=False):
        """Create an array mask representing the telescope pupil
        for a given image size and pixel scale.
        """

        ## Initial variables for pupil size and scale
        n = imShape[0]
        outD = self.outD
        inD = self.inD

        xgrid = np.zeros((n,n))

        ## Check if frequency scale flag is set
        if freqshift:
            for j in np.arange(n):
                xgrid[:,j] = j - (j > n/2)*n
        else:
            for j in np.arange(n):
                xgrid[:,j] = j
            if (n % 2):
                if whole:
                    offset = (n-1)/2.0
                else:
                    offset = 0.
            else:
                if whole:
                    offset = n/2.0
                else:
                    offset = (n-1)/2.0
            xgrid = xgrid - offset
    
        ## Construct grid of radial distances from center
        ax = xgrid * pixscale
        ay = ax.transpose()
        ar = np.sqrt(ax**2 + ay**2)

        ## Construct boolean array to characterize pupil
        ap_outer = (ar < outD/2)
        ap_inner = (ar < inD/2)        
        pupil = np.logical_not(ap_outer - ap_inner)

        ## Return boolean array for use as array mask
        return pupil

###############################################################################
##
## Telescope Dictionary
##
###############################################################################

TELESCOPE_DICT = {'LSST': Telescope('LSST', 8.36, 4.85),
                  'GPI': Telescope('GPI', 7.77, 1.024)}


###############################################################################
##
## Testing and Debug Code
##
###############################################################################

if __name__ == '__main__':

    ## Add debug code here
    print TELESCOPE_DICT['LSST']
