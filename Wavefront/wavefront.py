"""
Submodule containing main classes for wavefront phase analysis.

This module contains the main classes that are used to define wavefront phase
objects and perform simple Zernike decomposition analysis. 
"""


import numpy as np
import sys
import copy
import numpy.lib.index_tricks as itricks
from scipy import optimize, ndimage
from scipy.signal import periodogram
from astropy.io import fits


from zernike import Zernike
from telescope import Telescope, TELESCOPE_DICT
from waveplot import implot, Zplot, PSDplot

###############################################################################
##
## Image Class
##
###############################################################################

class Image():
    """Class representing an image. Attributes are data (array data) as a 
    NumPy array, and pixel scale (float pixscale)"""

    def __init__(self, data, pixscale=7.77/43):

        self.data = data
        self.pixscale = pixscale

    def __repr__(self):
        return '<{0}.{1} object at {2}>'.format(self.__module__, 
                                               type(self).__name__,
                                               hex(id(self)))

    def __str__(self):
        return '{0}x{1} Image; Pixel Scale {2}'.format(self.data.shape[0],
                                                       self.data.shape[1],
                                                       self.pixscale)

    @classmethod
    def import_fits(cls, file_path, pixscale=7.77/43):
        """Create an Image class from a FITs file. Can process a 2-D single 
        image data array or a 3-D image datacube.  Return is a single Image
        object or a list of Image objects, respectively."""

        ## Read fits file
        hdulist = fits.open(file_path, memmap=True)
        data = hdulist[0].data

        shape = data.shape

        ## Create Image objects
        if len(shape) == 2:
            return cls(data, pixscale)
        elif len(shape) == 3:
            image_list = []

            ## Iterate over datacube and initialize Image objects
            for i in range(data.shape[0]):
                single_image_data = data[i,:,:]
                image_list.append(cls(single_image_data, pixscale))  
            return image_list
        else:
            print shape
            sys.exit("FITs Read Error: Must be 2-D or 3-D Image datacube")

    def is_valid_mask(self, mask):
        """Check that provided mask matches the shape of the data."""

        shape = self.data.shape
        if mask is None:
            return True
        elif mask.shape == shape:
            return True
        else:
            print "Warning: Mask shape must be {}. No mask used".format(shape)
            return False     

    def export_fits(self, mask=None, **kwargs):
        """Export Image as a NumPy array to a FITs file."""
        
        ## Check key word arguments
        save_file = kwargs.pop('save_file', 'image.fits')
        fill_value = kwargs.pop('fill_value', 0.)

        ## Check if mask provided matches data shape
        if self.is_valid_mask(mask):
            masked_data = np.ma.MaskedArray(self.data, mask=mask, 
                                            fill_value=fill_value)
            masked_data = masked_data.filled()
        ## If not export data as is
        else:
            masked_data = self.data

        ## Export Image as FITs array
        hdu = fits.PrimaryHDU(masked_data)
        hdulist = fits.HDUList([hdu])
        hdulist.writeto(save_file, **kwargs)
        hdulist.close()

    def display(self, mask=None, **kwargs):
        """Display the Image and optionally save as a PNG file."""

        fill_value = kwargs.pop('fill_value', 0.)

        ## Check if mask provided matches data shape
        if self.is_valid_mask(mask):
            masked_data = np.ma.MaskedArray(self.data, mask=mask, 
                                            fill_value=fill_value)
        else:
            masked_data = self.data     

        ## Send data to image plotting function
        implot(masked_data, **kwargs)

    def zoom(self, zoom, order=3):
        """Resize array by zoom factor and perform interpolation."""

        ## Resize image and adjust pixel scale
        self.data = ndimage.zoom(self.data, zoom, order=order)
        self.pixscale = self.pixscale/float(zoom)

    def resample(self, outShape, outPixscale):
        """Resample input array to outShape and outPixscale specifications."""

        innRow, innCol = self.data.shape
        outnRow, outnCol = outShape

        ## Contruct map from outputRow to inputRow
        iRowBinEdges = np.zeros((innRow+1))
        oRowBinCenters = np.zeros((outnRow))

        iRowCenter = (innRow-1.)/2.
        for iRow in range(innRow+1):
            iRowBinEdges[iRow] = (iRow - iRowCenter - 0.4)*self.pixscale

        oRowCenter = (outnRow-1.)/2.
        for oRow in range(outnRow):
            oRowBinCenters[oRow] = (oRow - oRowCenter)*outPixscale

        ## find output centers in input bins
        oRowMap = np.digitize(oRowBinCenters, iRowBinEdges) - 1
    
        ## construct map from outputCol to inputCol
        iColBinEdges = np.zeros((innCol+1))
        oColBinCenters = np.zeros((outnCol))

        iColCenter = (innCol-1.)/2.
        for iCol in range(innCol+1):
            iColBinEdges[iCol] = (iCol - iColCenter - 0.5)*self.pixscale

        oColCenter = (outnCol-1.)/2.
        for oCol in range(outnCol):
            oColBinCenters[oCol] = (oCol - oColCenter)*outPixscale

        ## find output centers in input bins
        oColMap = np.digitize(oColBinCenters, iColBinEdges) - 1

        ## Perform resampling
        outArr = np.zeros((outnRow, outnCol))
        for orow in range(outnRow):
            for ocol in range(outnCol):
                irow = oRowMap[orow]
                icol = oColMap[ocol]
                if irow >= 0 and irow < innRow and icol >= 0 and icol < innCol:
                    outArr[orow,ocol] = self.data[irow,icol]

        self.data = outArr
        self.pixscale = outPixscale

    def mean_center(self, mask=None):
        """Subtract mean value from (masked) Image."""

        if self.is_valid_mask:
            masked_data = np.ma.MaskedArray(self.data, mask=mask, 
                                            fill_value=0.)
            avg = masked_data.mean(dtype=np.float64)
            self.data -= avg

    def apply_mask(self, mask, fill_value=0.):
        """Fill masked image data with fill value."""

        if self.is_valid_mask:
            masked_data = np.ma.MaskedArray(self.data, mask=mask, 
                                            fill_value=fill_value)
            self.data = masked_data.filled()

    def add_zernike(self, Z, a):
        """Modify image by adding a zernike polynomial."""

        ## Currently only works for 256x256, so resize image
        num_pix = self.data.shape[0]  
        self.zoom(256./num_pix)

        ## Get the desired zernike polynomial
        zernike_list = WaveFitter.get_zernikes(Z+1)
        zernike = zernike_list[Z,:,:]

        ## Combine images and resize to original size
        self.data = a*zernike + self.data
        self.zoom(num_pix/256.)

###############################################################################
##
## Wave Fitter Class
##
###############################################################################

class WaveFitter():
    """Class to control the fitting of zernike polynomials to wavefronts."""

    def __init__(self, mask, fill_value=0.):

        ## Check mask has dimensions 256x256
        if not self.is_valid_mask(mask):
            mask = None
        self._mask = mask
        self.fill_value = fill_value
        self._results = None
        self._models = None
        self._periodograms = None
        self._f = None
        self._is_processed = False
        self._is_finalized = False

    def __repr__(self):
        return '<{0}.{1} object at {2}>'.format(self.__module__, 
                                                type(self).__name__,
                                                hex(id(self)))

    def __str__(self):

        if self._is_finalized:
            return 'Finalized Wave Fitter with Results and Models'
        elif self._is_processed:
            return 'Processed Wave Fitter with Results'
        else:
            return 'Initialized Wave Fitter'  

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, new_mask):
        if self.is_valid_mask(new_mask):
            self._mask = new_mask

    @property
    def results(self): return self._results

    @property
    def periodograms(self): return self._periodograms

    @property
    def f(self): return self._f

    @property
    def models(self): return self._models

    @property
    def is_processed(self): return self._is_processed
 
    @property
    def is_finalized(self): return self._is_finalized

    @classmethod
    def make_model(cls, coefficients, nbin=256, zernikes=None, mask=None,
                   fill_value=0., as_image=False, pixscale=7.77/43, 
                   **kwargs):
        """Builds a zernike wavefront model from given coefficients.  
        Default return format is a masked array.
        """

        ## If zernikes aren't provided, generate them
        if zernikes is None:
            coefficients = np.array(coefficients)
            zernikes = cls.get_zernikes(len(coefficients), nbin, **kwargs)

        ## Find the zernikes, weighted by the coefficient values
        weighted_zernike = coefficients[:, None, None]*zernikes
        zernike_fit = weighted_zernike.sum(axis=0)

        ## Check if mask is provided
        if not cls.is_valid_mask(mask):   
            mask = None  
        zernike_fit = np.ma.MaskedArray(zernike_fit, mask=mask, 
                                        fill_value=fill_value)

        ## Return as image or numpy array
        if as_image:
            return Image(zernike_fit.data, pixscale)
        else:
            return zernike_fit

    @staticmethod
    def get_zernikes(nZ, nbin=48, **kwargs):
        """Create a list of zernike polynomials."""

        # build the x,y grid
        pupildiameter = kwargs.pop('pupildiameter', 8.0)
        radiusOuter = kwargs.pop('radiusOuter', 4.0)
        lo = -pupildiameter/2.
        hi = pupildiameter/2.
        yaxis, xaxis = itricks.mgrid[lo:hi:1j*nbin, lo:hi:1j*nbin]

        # build the rho, theta areas on the pupil
        rho = np.sqrt(xaxis*xaxis+yaxis*yaxis)/radiusOuter
        theta = np.arctan2(yaxis, xaxis)

        zpoly = Zernike(rho, theta, nZ)

        return zpoly.ZernikeTerm

    @staticmethod
    def is_valid_mask(mask):
        """Check that the mask is valid."""

        if mask is None:
            return True
        elif mask.shape[0] == mask.shape[1]:
            return True
        else:
            print "Mask array shape must be (X,X). No mask used."
            return False    

    def process(self, wavefront_list, nZ=37, **kwargs):
        """Run wavefront fit code on list of wavefronts and generate
        periodograms."""

        ## Check that list is provided
        if not isinstance(wavefront_list, list):
            wavefront_list = [wavefront_list]

        ## Check that mask is the correct size for the wavefronts
        if not wavefront_list[0].is_valid_mask(self._mask):
            print "Mask array shape must match wavefront. No mask used."
            self._mask = None

        ## Get zernike arrays
        nbin = wavefront_list[0].data.shape[0]
        zernikes = self.get_zernikes(nZ, nbin, **kwargs)
	zernike_list = []

	## Apply pupil mask and compress each array
	for i in range(zernikes.shape[0]):
	    masked_zernikes = np.ma.MaskedArray(zernikes[i,:,:], 
						mask=self._mask,
                                                fill_value=0.)
            zernike_list.append(masked_zernikes.compressed())

	## Create Z array for LLS fit
	Z = np.asarray(zernike_list).T

	## Perform LLS for each wavefront in the list
        results = []
        for wavefront in wavefront_list:

            ## Apply mask
            data = np.ma.MaskedArray(wavefront.data, 
                                     self._mask, 
                                     fill_value=0.)

	    b = data.compressed()
            fit_results = np.linalg.lstsq(Z, b)[0]
            
            results.append(fit_results)

        self._results = np.asarray(results).T

        ## Generate periodograms
        periodograms = []
        
        for i in range(self.results.shape[0]):
            f, Pxx_spec = periodogram(self.results[i,:], 1000.)
            periodograms.append(Pxx_spec)

        self._periodograms = np.asarray(periodograms)
        self._f = f

        ## Set flag for processed data
        self._is_processed = True

    def writePSDs(self, out_file_name):
        """Write periodogram outputs to CSV."""
        
        ## Append frequencies and save periodograms to CSV file
        if self._is_processed:

            output = np.vstack((self.f, self.periodograms))
            np.savetxt(out_file_name, output, delimiter=",")
        else:
            raise RuntimeError('WaveFitter models not processed!')     

    def finalize(self, pixscale=7.77/43):
        """Create zernike model for each wavefront object."""

        ## Create Zernike model for each image in time series
        for i in range(self.results.shape[1]):
            model = self.make_model(self.results[:,i], as_wave=True, pixscale=pixscale)
            self._models.append(model)     
 
        ## Set flag for finalized models
        self._is_finalized = True

    def clear(self):
        """Clear results and models"""
    
        self._results = None
        self._models = None
        self._periodograms = None
        self._f = None
        self._is_processed = False
        self._is_finalized = False     

    def write(self, out_file_name):
        """Write coefficient output to a csv file."""

        ## Save results to CSV file
        if self._is_processed: 
            np.savetxt(out_file_name, self.results, delimiter=",")    
        else:
            raise RuntimeError('WaveFitter models not processed!')

    def saveZgraphs(self, filebase):
        """Save graph of time series plot as PNG for each Zernike"""
        
        ## Generate time series plot for each Zernike
        if self._is_processed:

            for i in range(self.results.shape[0]):
                Zplot(self.results[i,:], i+1, filebase)
        else:
            raise RuntimeError('WaveFitter models not processed!')

    def savePSDgraphs(self, filebase):

        if self._is_processed:

            slopes = []
            intercepts = []
            
            for i in range(self.periodograms.shape[0]):
                slope, intercept = PSDplot(self.periodograms[i,:], 
                                                    self.f, i+1, filebase)
                slopes.append(slope)
                intercepts.append(intercept)

            powerlawfitfile = '{0}_PSD_fit.csv'.format(filebase)
            np.savetxt(powerlawfitfile, [slopes, intercepts], delimiter=',')
        else:
            raise RuntimeError('WaveFitter models not processed!')

    def export_models(self, out_file_name='models.fits', 
                      is_masked=False, **kwargs):
        """Export images of the models to a FITs datacube."""
        
        ## Check finalized flag
        if self._is_finalized:

            ## Check mask flag and create data cube to export
            if is_masked:
                models = self._models
                for model in models:
                    model.apply_mask(self._mask)
                model_images = [model.data for model in models]
            else:
                model_images = [model.data for model in self._models]

            hdu1 = fits.HDUList()
            hdu1.append(fits.PrimaryHDU())
        
            for image in model_images:
                hdu1.append(fits.ImageHDU(data=img))

            hdu1.writeto(out_file_name, **kwargs)

        else:
            raise RuntimeError("WaveFitter models not processed!")

###########################################################################
##
## Testing and Debug Code
##
###########################################################################

if __name__ == '__main__':

    ## Add debug code here

    ## Debug new plotting code
    data = np.random.randn(48, 48)
    test = Image(data)
    test.display()
