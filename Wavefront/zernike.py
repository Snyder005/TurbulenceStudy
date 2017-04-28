#
# $Rev:: 28            $:  
# $Author:: roodman    $:  
# $LastChangedDate:: 2#$:  
#
#
# Zernike class
#

import numpy
from scipy.misc import factorial
from scipy import linalg
import pdb

class Zernike(object):
    """ zernike stores an array of all Zernike polynomials up to 
    the desired order.  It uses hand coded terms to highly optimize 
    the calculation
    """
    
    def __init__(self,rhoArr,thetaArr,nTerms=37,printFlag=0):
        """ initialize Zernike class here, and fill Zernike array
        """
        self.rho = rhoArr.copy()
        self.theta = thetaArr.copy()
        if nTerms<3:
            self.nTerms = 3
        else:
            self.nTerms = nTerms
        self.nBins = rhoArr.shape[0]
        self.ZernikeTerm = numpy.zeros((self.nTerms,self.nBins,self.nBins))        
        self.printFlag = printFlag
        self.calcEmAll()

        self.ZernikeDescription = ["Piston (Bias)          1",
                                   "Tilt X                 4^(1/2) (p) * COS (A)",
                                   "Tilt Y                 4^(1/2) (p) * SIN (A)",
                                   "Power (Defocus)        3^(1/2) (2p^2 - 1)",
                                   "Astigmatism Y          6^(1/2) (p^2) * SIN (2A)",
                                   "Astigmatism X          6^(1/2) (p^2) * COS (2A)",
                                   "Coma Y                 8^(1/2) (3p^3 - 2p) * SIN (A)",
                                   "Coma X                 8^(1/2) (3p^3 - 2p) * COS (A)",
                                   "Trefoil Y              8^(1/2) (p^3) * SIN (3A)",
                                   "Trefoil X              8^(1/2) (p^3) * COS (3A)",
                                   "Primary Spherical      5^(1/2) (6p^4 - 6p^2 + 1)",
                                   "Secondary Astig X      10^(1/2) (4p^4 - 3p^2) * COS (2A)",
                                   "Secondary Astig Y      10^(1/2) (4p^4 - 3p^2) * SIN (2A)",
                                   "TetraFoil X            10^(1/2) (p^4) * COS (4A)",
                                   "TetraFoil Y            10^(1/2) (p^4) * SIN (4A)",
                                   "Secondary Coma X       12^(1/2) (10p^5 - 12p^3 + 3p) * COS (A)",
                                   "Secondary Coma Y       12^(1/2) (10p^5 - 12p^3 + 3p) * SIN (A)",
                                   "Secondary Trefoil X    12^(1/2) (5p^5 - 4p^3) * COS (3A)",
                                   "Secondary Trefoil Y    12^(1/2) (5p^5 - 4p^3) * SIN (3A)",
                                   "Pentafoil X            12^(1/2) (p^5) * COS (5A)",
                                   "Pentafoil Y            12^(1/2) (p^5) * SIN (5A)",
                                   "Secondary Spherical    7^(1/2) (20p^6 - 30p^4 + 12p^2 - 1)",
                                   "Tertiary Astig Y       14^(1/2) (15p^6 - 20p^4 + 6p^2) * SIN (2A)",
                                   "Tertiary Astig X       14^(1/2) (15p^6 - 20p^4 + 6p^2) * COS (2A)",
                                   "Secondary Tetrafoil Y  14^(1/2) (6p^6 - 5p^4) * SIN (4A)",
                                   "Secondary Tetrafoil X  14^(1/2) (6p^6 - 5p^4) * COS (4A)",
                                   "Sextafoil Y            14^(1/2) (p^6) * SIN (6A)",
                                   "Sextafoil X            14^(1/2) (p^6) * COS (6A)",
                                   "Tertiary Coma Y        16^(1/2) (35p^7 - 60p^5 + 30p^3 - 4p) * SIN (A)",
                                   "Tertiary Coma X        16^(1/2) (35p^7 - 60p^5 + 30p^3 - 4p) * COS (A)",
                                   "Tertiary Trefoil Y     16^(1/2) (21p^7 - 30p^5 + 10p^3) * SIN (3A)",
                                   "Tertiary Trefoil X     16^(1/2) (21p^7 - 30p^5 + 10p^3) * COS (3A)",
                                   "Secondary Pentafoil Y  16^(1/2) (7p^7 - 6p^5) * SIN (5A)",
                                   "Secondary Pentafoil X  16^(1/2) (7p^7 - 6p^5) * COS (5A)",
                                   "Septafoil Y            16^(1/2) (p^7) * SIN (7A)",
                                   "Septafoil X            16^(1/2) (p^7) * COS (7A)",
                                   "Tertiary Spherical     9^(1/2) (70p^8 - 140p^6 + 90p^4 - 20p^2 + 1)"]
        
        if self.nTerms>37:
            for iZ in range(37,self.nTerms):
                self.ZernikeDescription.append(" ")



    def calcEmAll(self):

        # first figure out what nMax is for nTerms
        numZ = 0
        nLast = -1
        #pdb.set_trace()
        while numZ<self.nTerms :
            nLast = nLast + 1
            for m in range(nLast+1):
                if numpy.mod(nLast-m,2)==0 :
                    if m==0:
                        numZ = numZ + 1
                    else:
                        numZ = numZ + 2

        nRhoPowers = nLast + 1
        nTrigTerms = nLast + 1

        # first fill rho^n terms, cos(n theta) and sin(n theta)

        #
        # indexing is self.rhoPowersArr[0] is r^0
        #                              [1] is r^1  etc...
        #
        rhoNArr = numpy.zeros((nRhoPowers,self.nBins,self.nBins))

        rhoNArr[0] = numpy.ones((self.nBins,self.nBins))
        rhoNArr[1] = self.rho
        for iPower in range(2, nRhoPowers,1):
            rhoNArr[iPower] = rhoNArr[iPower-1] * self.rho

        #
        # self.cosTheta[0] = is just 1
        # self.cosTheta[1] = cos(theta)
        # self.cosTheta[2] = cos(2 theta)  etc...
        #
        cosNArr = numpy.zeros(( nTrigTerms, self.nBins, self.nBins))
        sinNArr = numpy.zeros(( nTrigTerms, self.nBins, self.nBins))
        
        cosNArr[0] = numpy.ones(( self.nBins, self.nBins))
        cosNArr[1] = numpy.cos( self.theta)
        sinNArr[0] = numpy.zeros(( self.nBins, self.nBins))
        sinNArr[1] = numpy.sin( self.theta)
        
        for iTrigTerm in range(2, nTrigTerms,1):
#             cosNArr[iTrigTerm] = cosNArr[iTrigTerm-1]*cosNArr[1] - sinNArr[iTrigTerm-1]*sinNArr[1]
#             sinNArr[iTrigTerm] = sinNArr[1]*cosNArr[iTrigTerm-1] + sinNArr[iTrigTerm-1]*cosNArr[1]
            cosNArr[iTrigTerm] = numpy.cos( iTrigTerm * self.theta )
            sinNArr[iTrigTerm] = numpy.sin( iTrigTerm * self.theta )


# Trig formula:
# cos(t+t') = costcost' - sintsint'
# cos3t = cos(2t+t) = cos2tcost-sin2tsint
# cosNt = cos([N-1]t+t) = cos[N-1]tcost-sin[N-1]tsint
#
# sin(t+t') = sintcost' + sint'cost
# sin(3t) = sintcos2t + sin2tcost
# sin(Nt) = sintcos(N-1)t + sin(N-1)tcost


        #
        # now calculate the Zernike polynomials
        #

        # store indicies etc., access by iZ+1 not iZ, ie. the actual Zernike order number !!!!
        self.n = numpy.zeros((self.nTerms+1))
        self.m = numpy.zeros((self.nTerms+1))
        self.coeff = numpy.zeros((self.nTerms+1))
        self.sinorcos = numpy.zeros((self.nTerms+1))

        iZ = -1   # Zernike term counter
        for n in range(nRhoPowers):
            for m in range(n+1):
                if numpy.mod(n-m,2)==0 :

                    nn = (n-m)/2
                    nm = (n+m)/2

                    # radial term
                    radialTerm = numpy.zeros((self.nBins,self.nBins))
                    for s in range(nn+1):
                        rcoeff = numpy.power(-1,s) * factorial(n-s) / (  factorial(s) * factorial(nm-s) * factorial(nn-s)   )
                        radialTerm = radialTerm + rcoeff * rhoNArr[n-2*s]

                    coeff = numpy.sqrt(2.*n+2.)
                    if m==0:
                        coeff = coeff/numpy.sqrt(2.)

                    # even and odd
                    if m==0:
                        iZ = iZ + 1
                        if (iZ<self.nTerms):
                            self.ZernikeTerm[iZ] = coeff * radialTerm 
                        if self.printFlag>0:
                            print "zernike:: ",iZ+1,n,m,coeff
                            self.n[iZ+1] = n
                            self.m[iZ+1] = m
                            self.coeff[iZ+1] = coeff
                            self.sinorcos[iZ+1] = 0


                    # convention in Noll is that iZ odd is sin, and iZ even is cos
                    # but Noll also has iZ starting from 1, but we start at 0, so it is reversed!

                    else:  
                        iZ = iZ + 1
                        if (numpy.mod(iZ,2)!=0):
                            if (iZ<self.nTerms):
                                self.ZernikeTerm[iZ] = coeff * radialTerm * cosNArr[m] 
                                if self.printFlag>0:
                                    print "zernike:: ",iZ+1,n,m,coeff," cos"
                                    self.n[iZ+1] = n
                                    self.m[iZ+1] = m
                                    self.coeff[iZ+1] = coeff
                                    self.sinorcos[iZ+1] = 1


                            iZ = iZ + 1
                            if (iZ<self.nTerms):
                                self.ZernikeTerm[iZ] = coeff * radialTerm * sinNArr[m] 
                                if self.printFlag>0:
                                    print "zernike:: ",iZ+1,n,m,coeff," sin"
                                    self.n[iZ+1] = n
                                    self.m[iZ+1] = m
                                    self.coeff[iZ+1] = coeff
                                    self.sinorcos[iZ+1] = -1

                        else:
                            if (iZ<self.nTerms):
                                self.ZernikeTerm[iZ] = coeff * radialTerm * sinNArr[m] 
                                if self.printFlag>0:
                                    print "zernike:: ",iZ+1,n,m,coeff," sin"
                                    self.n[iZ+1] = n
                                    self.m[iZ+1] = m
                                    self.coeff[iZ+1] = coeff
                                    self.sinorcos[iZ+1] = -1

                            iZ = iZ + 1
                            if (iZ<self.nTerms):
                                self.ZernikeTerm[iZ] = coeff * radialTerm * cosNArr[m] 
                                if self.printFlag>0:
                                    print "zernike:: ",iZ+1,n,m,coeff," cos"
                                    self.n[iZ+1] = n
                                    self.m[iZ+1] = m
                                    self.coeff[iZ+1] = coeff
                                    self.sinorcos[iZ+1] = 1




    def calcDiagonalBasis(self,pupilMask,zernikeList,inputZernikes):

        # calculate Orthogonal Basis of Zernike polynomials on the supplied pupilMask
        # only use desired list of zernike's, output list to have same dimensionality

        area = 1.0/pupilMask.sum()
        ndim = len(zernikeList)

        self.A = numpy.zeros((ndim,ndim))
        for i in range(ndim):
            iZ = zernikeList[i] 
            for j in range(ndim):
                jZ = zernikeList[j]
                self.A[i,j] = (area * pupilMask * inputZernikes[iZ] * inputZernikes[jZ]).sum()
                
        # apply a cutoff to A
        aboveThresh = numpy.where(numpy.abs(self.A)>1e-8,1.0,0.0)
        self.A = self.A * aboveThresh
        
        # diagonalize A and find eigenvectors
        eigenval,self.eigenvec = linalg.eigh(self.A)
        # save transpose of eigenvec matrix for use in finding fit coefficients in
        # the original basis
        self.mortho = numpy.matrix(self.eigenvec)
        self.morthot = self.mortho.transpose()

        # apply a cutoff to eigenvec
        aboveThresh = numpy.where(numpy.abs(self.eigenvec)>1e-8,1.0,0.0)
        self.eigenvec = self.eigenvec * aboveThresh

        # construct primed Zernike expansion - now diagonal
        outputZernikes = numpy.zeros((ndim,self.nBins,self.nBins))

        for i in range(ndim):
            eigv = self.eigenvec[:,i]
            for j in range(ndim):
                jZ = zernikeList[j]
                outputZernikes[i] = outputZernikes[i] + inputZernikes[jZ]*eigv[j]

        return outputZernikes
                    
    def getDiagonalBasis(self,pupilMask,zernikeList):

        self.ZernikePrimeTerm = self.calcDiagonalBasis(pupilMask,zernikeList,self.ZernikeTerm)

    def toPrimeBasis(self,regularArray):
        regularMatrix = numpy.matrix(regularArray).transpose()
        primeMatrix = self.morthot * regularMatrix
        pArray = numpy.array(primeMatrix.transpose())
        primeArray = pArray[0]
        return primeArray

    def toRegularBasis(self,primeArray):
        primeMatrix = numpy.matrix(primeArray).transpose()
        regularMatrix = self.mortho * primeMatrix
        rArray = numpy.array(regularMatrix.transpose())
        regularArray = rArray[0]
        return regularArray

    
