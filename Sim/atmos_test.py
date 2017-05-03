import subprocess
import numpy as np

## Generate parameter space of r0s and velocities to test
r0s = np.linspace(0.1, 0.2, num=10)
vels = np.linspace(10.0, 20.0, num=10)

for i in range(r0s.shape[0]):
    for j in range(vels.shape[0]):

        result = subprocess.check_output(['python', 'make_ar_atmos_test.py', 
                                          '15', '1000', '0.999', '48', '1', 
                                          '-v', "{0:.2f}".format(vels[j]), 
                                          '-r', "{0:.2f}".format(r0s[i])])

        print result



 
    
