"""
Main module for wavefront phase data analysis.

This serves as the main module for the wavefront phase data analysis package. It serves to import the most used classes from the submodules.

Author: Adam Snyder
"""


__version__ = 4.1

from wavefront import Image
from wavefront import WaveFitter
from telescope import Telescope
from telescope import TELESCOPE_DICT

__all__ = ['wavefront', 'zernike', 'telescope', 'waveplot']
