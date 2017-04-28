from os.path import join, splitext
from wavefront import Image
import os
import numpy as np
import treecorr

###############################################################################
##
## Utils
##
###############################################################################

def image_to_catalog(image_list, mask=None, is_centered=True):
    """Converts a list of images to a catalog.
       Currently assumed that all images in the list are the same size"""

    ## Check if single wavefront or list provided
    if not isinstance (image_list,list): image_list = [image_list]
    if len(image_list) == 0:
        raise ValueError("No images provided")

    shape = image_list[0].data.shape
    nX = shape[0]
    nY = shape[1]
    catalog_list = []

    ## Check if mask is provided (may put more mask checks here)
    if mask is None: mask=make_mask_none(shape)

    ## Create catalog for each image and add to list
    for image in image_list:

        ## Check mean centered flag
        if is_centered: image.mean_center(mask=mask)

        x = []
        y = []
        w = []

        for i in range(nX):
            for j in range(nY):
                if not mask[i,j]:
                    x.append(float(i))
                    y.append(float(j))
                    w.append(image.data[i,j])

        x = np.array(x)
        y = np.array(y)
        w = np.array(w)

        catalog_list.append(treecorr.Catalog(x=x, y=y, k=w))

    return catalog_list 

def make_dir(directory_path):

    try:
        os.makedirs(directory_path)
        print "Created directory at: ", directory_path
    except OSError:
        if not os.path.isdir(directory_path):
            raise

def filepath_library(data_file, main_directory, telescope_name):
    """Creates necessary directories and filepaths for graphs and results.
    
       Create a """

    data_filepath = join(main_directory, 'Data', data_file)
    base = '{0}_{1}'.format(splitext(data_file)[0], telescope_name)

    ## Create main results directory
    results_dir = join(main_directory, 'Results', base)
    make_dir(results_dir)

    ## Create Graphs directory
    graphs_dir = join(results_dir, 'Graphs')
    make_dir(graphs_dir)

    ## Construct filepaths for all output files

    zernikes = join(results_dir, '{0}_coefficients.csv'.format(base))
    periodograms = join(results_dir, '{0}_periodograms.csv'.format(base))
    graphs = join(graphs_dir, base)
 
    ## Make dictionary to hold all the necessary filepaths
    filepath_dict = {'DataDir' : data_filepath,
                     'Base' : base, 
                     'zernikes' : zernikes,
                     'periodograms' : periodograms,
                     'graphs' : graphs,
                     'GraphsDir' : graphs_dir,
                     'ResultsDir' : results_dir}

    return filepath_dict

def powerlaw(C, r):

    y = C[0] + C[1]*(r**C[2])
    return y

def residuals(C, y, r):

    err = y - powerlaw(C, r)
    return err

###############################################################################
##
## Debug Code
##
###############################################################################

if __name__ == '__main__':

    ## Debug code here
    pass
