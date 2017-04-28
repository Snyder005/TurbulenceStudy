import subprocess
from astropy.io import fits
import os

base_directory = '/nfs/slac/g/ki/ki19/lsst/snyder18/Data'

with open('phasefiles_filtered.txt', 'r') as f:
    for line in f:
        
        phasefile = "aored_{0}_poldm_phase.fits".format(line.rstrip())

        subprocess.Popen(['bsub', '-W', '4:00', '-o', 'lots.log', 'python', 
                          'full_wavefront.py', 'GPI', phasefile])

        print "Process submitted."
        print phasefile, '\n'

    
