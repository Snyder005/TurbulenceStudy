#!/usr/bin/env python
"""
Script to process a single GPI phase data file.

This script performs a series of basic data analysis functions on a reduced GPI 
phase data file.  A Zernike decomposition is performed on the masked data cube
and set of time series and PSDs are generated and saved to disk, along with the
corresponding images of the graphs.

Todo:
    * Rewrite the subdirectory tree creation to be more flexible.
"""

###############################################################################

import os
import argparse
import sys
import numpy as np
import copy

from os.path import join, splitext
from scipy.signal import periodogram

from Wavefront import Telescope
from Wavefront import Image
from Wavefront import WaveFitter
from Wavefront import TELESCOPE_DICT
from Wavefront import waveplot, utils

def main(datafile, telescope_name, directory, save_images=True):
    """Perform Zernike decomposition and save data products to a specific
    subdirectory tree.

    Args:
        datafile (str): GPI reduced data file name.
        telescope_name (str): Name of built-in telescope parameters to use.
        directory (str): Name of main directory to build subdirectory tree in.
        save_images (bool): Specify whether to save graph images.
    """

    ## File path dictionary
    file_dict = utils.filepath_library(datafile, directory, telescope_name)

    ## Get telescope and wavefront time series
    telescope = TELESCOPE_DICT[telescope_name]
    pupil_mask = telescope.get_pupil((48,48), pixscale=7.77/43)
    wavefront_list = Image.import_fits(file_dict['DataDir'], pixscale=7.77/43)

    ## Initialize fitter and process wavefronts
    fitter = WaveFitter(mask=pupil_mask, fill_value=0.)
    fitter.process(wavefront_list)

    ## Save Zernike time series and periodograms
    fitter.write(file_dict['zernikes'])
    fitter.writePSDs(file_dict['periodograms'])

    ## Save graphs of Zernike time series and periodograms
    fitter.saveZgraphs(file_dict['graphs'])
    fitter.savePSDgraphs(file_dict['graphs'])
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("telescope", default='GPI', 
                        help='Built-in telescope parameters to use')
    parser.add_argument("filename", help='Name of data file.')
    parser.add_argument("directory", help='Main directory to perform all analysis')

    args = parser.parse_args()
    directory = args.directory
    filename = args.filename
    if args.telescope in TELESCOPE_DICT:
        telescope_name = args.telescope
    else:
        telescope_name = 'GPI'

    main(filename, telescope_name, directory)

