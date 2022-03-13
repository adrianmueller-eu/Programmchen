import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import polyfit, polyval

# TODO: exponential fit
# TODO: log fit
# TODO: sin, cos
# TODO: arbitrary kernel

def pm(x, y, deg, plot=True):
    coeff = polyfit(x,y,deg)
    poly = Polynomial(coeff)
    if plot:
        ax = poly.plot(x)
        ax.scatter(x,y)
        plt.show()
    return poly
    #return lambda x0: polyval(np.array(x0), coeff)

def lm(x,y, plot=True):
    return pm(x,y,1, plot)

class Polynomial:

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

    def plot(self, x, ax=None):
        if ax is None:
            ax = plt.gca()
        ax.plot(x, np.sum([c*x**i for i,c in enumerate(self.coeff)], axis=0),
                     label=" + ".join(["%.3fx^%d" % (c, i) for i,c in enumerate(self.coeff)]))
        ax.legend()
        return ax
