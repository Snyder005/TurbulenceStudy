#!/usr/bin/env python

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
    parser.add_argument("telescope", default='GPI')
    parser.add_argument("filename")
    parser.add_argument("directory")

    args = parser.parse_args()
    directory = args.directory
    filename = args.filename
    if args.telescope in TELESCOPE_DICT:
        telescope_name = args.telescope
    else:
        telescope_name = 'GPI'

    main(filename, telescope_name, directory)

