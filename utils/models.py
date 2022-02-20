import numpy as np
from numpy.polynomial.polynomial import polyfit, polyval

def pm(x,y,deg=1):

    coeff = polyfit(x,y,deg)
    return PolyFit(coeff)
    #return lambda x0: polyval(np.array(x0), coeff)

class PolyFit:

    def __init__(self, coeff):
        self.coeff = coeff
        pass

    @property
    def degree(self):
        return len(self.coeff)-1


    def __call__(self, x0):
        return polyval(np.array(x0), self.coeff)

    def __str__(self):
        return f"Polynomial of degree %d with coeff %s" % (self.degree, self.coeff)

    def __repr__(self):
        return self.__str__()
