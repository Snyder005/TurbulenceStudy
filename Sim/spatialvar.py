import numpy as np
import sys
from astropy.io import fits
import matplotlib.pyplot as plt
import argparse
import os
from Wavefront import TELESCOPE_DICT, Telescope, Image

def main(file_name, directory, telescope_name):

    filepath = os.path.join(directory, 'Data', file_name)

    ## Get telescope pupil
    telescope = TELESCOPE_DICT[telescope_name]
    pupil_mask = telescope.get_pupil((48, 48), pix_scale=7.77/43)

    hdulist = fits.open(filepath, memmap=True)
    data = hdulist[0].data

    ## Calculate variance map and apply pupil
    var_map = data.var(axis=0)

    ## Mean center and repeat calculations
    x = np.ones(data.shape[0])
    mask = x[:,None,None]*pupil_mask

    masked_data = np.ma.MaskedArray(data, mask=mask)

    mean_data = masked_data.mean(axis=2).mean(axis=1)

    mean_centered = masked_data - mean_data[:, None, None]
    mean_centered = mean_centered.data

    var_map_mc = mean_centered.var(axis=0)
    plt.imshow(var_map_mc, origin='lower')
    plt.colorbar()
    plt.show()

    plt.imshow(var_map)
    plt.show()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("telescope", default=0)
    parser.add_argument("file_name")
    parser.add_argument("-d", "--directory", default='/nfs/slac/g/ki/ki19/lsst/snyder18')
    args = parser.parse_args()

    directory = args.directory
    file_name = args.file_name
    telescope_dict = {'0' : 'GPI', '1' : 'LSST'}
    telescope_name = telescope_dict[args.telescope]

    main(file_name, directory, telescope_name)
