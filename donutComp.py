#!/usr/bin/env python

from os.path import join, splitext
import argparse
import numpy as np
from donutlib.makedonut import makedonut
import matplotlib.pyplot as plt
from astropy.io import fits

def donutcomp(coefficients, T=15000, output='example'):

    nT, nZ = coefficients.shape

    ## Initialize the input dictionary
    z4 = 10.
    z5 = 0.2
    z6 = 0.2
    inputDict = {"iTelescope":0,
                          "nZernikeTerms":37,
                          "nbin":512,
                          "nPixels":64,
                          "pixelOverSample":8,
                          "scaleFactor":1.,                 
                          "rzero":0.125,
                          "nEle":1.0e6,
                          "background":0.,
                          "randomFlag":False,
                          "randomSeed":209823,
                          "ZernikeArray":[0.,0.,0.,z4,z5,z6]}

    m = makedonut(**inputDict)

    ## Set number of individual phase maps to use

    if T > nT:
        return

    ## Calculate how many individual defocused images can be made
    num_img = nT/T

    for i in range(num_img):

        ## Average zernike coefficients and create defocused image
        avg_coefficients = coefficients[i*T:(i+1)*T, :].mean(axis=0, dtype=np.float64)
        avg_coefficients[3] += 10.
        ZDict = {"ZernikeArray": list(avg_coefficients/0.7),
                 "nEle" : 1.0e6,
                 "rzero" : 0.15}
        avg_donut = m.make(**ZDict)
    
        ## Export output to fits file
        hdu = fits.PrimaryHDU(avg_donut)
        hdulist = fits.HDUList([hdu])
        hdulist.writeto("{0}_avgdonut{1}_T{2}.fits".format(output, i, T), clobber=True)
        hdulist.close()

        ## Create defocused image for phase map and add

        donuts = []
        for j in range(T):
            coefficients[j+i*T, 3] += 10.
            ZDict = {"ZernikeArray": list(coefficients[j+i*T,:]/0.7),
                     "nEle" : 1.0e6,
                     "rzero": 1.0}
            donuts.append(m.make(**ZDict))

        donuts = np.array(donuts)
        comp_donut = donuts.mean(axis=0)

        ## Export output to fits file
        hdu = fits.PrimaryHDU(comp_donut)
        hdulist = fits.HDUList([hdu])
        hdulist.writeto("{0}_compdonut{1}_T{2}.fits".format(output, i, T), clobber=True)
        hdulist.close()

    else:
        
        ## Average zernike coefficients and create defocused image
        avg_coefficients = coefficients[num_img*T:, :].mean(axis=0, dtype=np.float64)
        avg_coefficients[3] += 10.
        ZDict = {"ZernikeArray": list(avg_coefficients/0.7),
                 "nEle" : 1.0e6,
                 "rzero" : 0.15}
        avg_donut = m.make(**ZDict)
    
        ## Export output to fits file
        hdu = fits.PrimaryHDU(avg_donut)
        hdulist = fits.HDUList([hdu])
        hdulist.writeto("{0}_avgdonut{1}_T{2}.fits".format(output, num_img, nT-num_img*T), clobber=True)
        hdulist.close()

        ## Create defocused image for phase map and add

        donuts = []
        for j in range(num_img*T, nT):
            coefficients[j, 3] += 10.
            ZDict = {"ZernikeArray": list(coefficients[j,:]/0.7),
                     "nEle" : 1.0e6,
                     "rzero": 1.0}
            donuts.append(m.make(**ZDict))

        donuts = np.array(donuts)
        comp_donut = donuts.mean(axis=0)

        ## Export output to fits file
        hdu = fits.PrimaryHDU(comp_donut)
        hdulist = fits.HDUList([hdu])
        hdulist.writeto("{0}_compdonut{1}_T{2}.fits".format(output, num_img, nT-num_img*T), clobber=True)
        hdulist.close()

def main(Zfile, T=15000, file_name='example'):

    ## Read zernike coefficients from file    
    coefficients = np.loadtxt(Zfile, dtype=np.float64, delimiter=",")
    donutcomp(coefficients, T=T, output=file_name)

if __name__ == '__main__':

    ## Create argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("telescope", default=0)
    parser.add_argument("file_name")
    parser.add_argument("-d", "--directory", default='/nfs/slac/g/ki/ki19/lsst/snyder18')
    parser.add_argument("-t", "--time", default=15000, type=int)
    args = parser.parse_args()

    ## Get arguments
    directory = args.directory
    integration_time = args.time
    telescope_dict = {'0' : 'GPI', '1' : 'LSST'}
    telescope_name = telescope_dict[args.telescope]
    file_name = join('{0}_{1}'.format(args.file_name, telescope_name))

    ## Construct filepath for coefficient file
    Zfile = '{0}_coefficients.csv'.format(file_name)
    Zfilepath = join(directory, 'Results', file_name, Zfile)

    main(Zfilepath, integration_time, file_name)
