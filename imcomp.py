from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import sys

def main(file1, file2):

    ## Read fits file
    hdulist1 = fits.open(file1, memmap=True)
    image1 = hdulist1[0].data

    hdulist2 = fits.open(file2, memmap=True)
    image2 = hdulist2[0].data

    residual = image1 - image2

    vmax = max(image1.max(), image2.max())

    ## Plot comparison of the two images
    ## Plot original wavefront, model wavefront, and residual
    fig, axes = plt.subplots(nrows=1, ncols=3)
    fig.suptitle("Defocused Images from Zernike Fits", fontsize=20)

    img0 = axes[0].imshow(image1, vmin=0, vmax=vmax, origin='lower')
    axes[0].set_title("Composite")
    cbar0 = plt.colorbar(img0, ax=axes[0], orientation='horizontal')

    img1 = axes[1].imshow(image2, vmin=0, vmax=vmax, origin='lower')
    axes[1].set_title("Average")
    cbar1 = plt.colorbar(img1, ax=axes[1], orientation='horizontal')
#    axes[1].co

    img2 = axes[2].imshow(residual, origin='lower')
    axes[2].set_title("Residual")
    cbar2 = plt.colorbar(img2, ax=axes[2], orientation='horizontal')
#    axes[2].colorbar()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':

    main(sys.argv[1], sys.argv[2])
