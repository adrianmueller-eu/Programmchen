import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import polyfit, polyval
from abc import ABC, abstractmethod

# TODO: exponential fit
# TODO: log fit
# TODO: sin, cos
# TODO: arbitrary kernel

# convenience functions
def pm(x, y, deg, plot=True):
    x = np.array(list(x))
    y = np.array(list(y))
    if deg >= 0:
        coeff = polyfit(x,y,deg)
        poly = Polynomial(coeff)
    else:
        coeff = polyfit(x,1/y,-deg)
        poly = InversePolynomial(coeff)
    if plot:
        ax = poly.plot(x)
        ax.scatter(x,y)
        #plt.show()
    return poly
    #return lambda x0: polyval(np.array(x0), coeff)

def lm(x, y, plot=True):
    return pm(x,y,1, plot)

# helper methods
def _generate_poly_label(coeff):
    q = len(coeff) - 1
    res = ""
    for i,c in enumerate(reversed(coeff)):
       if q-i == 0:
          res += "%.3f" % c
       else:
          res += "%.3fx^%d + " % (c, q-i)
    return res

# Functions
class Function(ABC):

    @abstractmethod
    def __call__(self, x):
        pass

    @abstractmethod
    def __str__(self):
        pass

    def __repr__(self):
        return self.__str__()

    @abstractmethod
    def _plot_label(self):
        pass

    def plot(self, x, ax=None):
        if ax is None:
            ax = plt.gca()
        ax.plot(x, self(x), label=self._plot_label())
        ax.legend()
        return ax


class Polynomial(Function):

    def __init__(self, coeff):
        self.coeff = coeff

    @property
    def degree(self):
        return len(self.coeff)-1

    def __call__(self, x):
        return polyval(np.array(x), self.coeff)

    def __str__(self):
        return f"Polynomial of degree %d with coeff %s" % (self.degree, self.coeff)

    def _plot_label(self):
        return _generate_poly_label(self.coeff)

class InversePolynomial(Function):

    def __init__(self, coeff):
        self.coeff = coeff

    @property
    def degree(self):
        return 1-len(self.coeff)

    def __call__(self, x):
        return 1/polyval(np.array(x), self.coeff)

    def __str__(self):
        return f"Inverse polynomial of degree %d with coeff %s" % (self.degree, self.coeff)

    def _plot_label(self):
        return "1/(" + _generate_poly_label(self.coeff) + ")"
