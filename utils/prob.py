import matplotlib.pyplot as plt
import numpy as np
from math import factorial
from .utils import *

def smooth(x, y, smoothing=0.1):
    # https://scipy.github.io/old-wiki/pages/Cookbook/SavitzkyGolay
    def savitzky_golay(y, window_size, order, deriv=0, rate=1):
        if window_size < order + 2:
            raise TypeError("window_size is too small for the polynomials order")
        order_range = range(order+1)
        half_window = (window_size -1) // 2
        # precompute coefficients
        b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
        m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
        # pad the signal at the extremes with
        # values taken from the signal itself
        firstvals = y[0] - np.abs(y[1:half_window+1][::-1] - y[0])
        lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
        y = np.concatenate((firstvals, y, lastvals))
        return np.convolve(m[::-1], y, mode='valid')

    # wilde guesses
    window = max(2,int(smoothing*len(x)))
    order = min(window-2,3)

    y = savitzky_golay(y, window, order)
    return x,y

# converts 1-d data into a pdf, smoothing in [0,1]
def density(data, plot=False, label=None, smoothing=0.1, log=False, num_bins=None):
    if log:
        bins = logbins(data, scale=2, num=num_bins+1 if num_bins else bins_sqrt(data)+1)
        n, bin_edges = np.histogram(data, bins=bins, density=True)
        bin_centers = 10**(moving_avg(np.log10(bin_edges), 2))
    else:
        bins = num_bins or bins_sqrt(data)
        bins += 1
        n, bin_edges = np.histogram(data, bins=bins, density=True)
        bin_centers = moving_avg(bin_edges, 2)

    if smoothing:
        x, y = smooth(bin_centers, n, smoothing)
        # normalization
        dx = np.diff(bin_edges)
        y /= np.sum(y*dx)
    else:
        x,y = bin_centers, n

    if plot:
        plt.figure(figsize=(10,5))
        plt.plot(x, y, label=label)
        top = max(plt.gca().get_ylim()[1], 1.05*np.max(y))
        plt.ylim(bottom=0, top=top)
        ax = plt.gca()
        ax.set_ylabel("Pdf")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        if log:
            plt.xscale('log')
        if label:
            plt.legend()

    return x, y

# x,y should be an output of "density" above
def resample(x, y, sample_size=int(1e6)):
    dx = x[1] - x[0]
    y_cs = np.cumsum(y*dx)
    from scipy.interpolate import interp1d
    f = interp1d(y_cs, x, bounds_error=False, fill_value=0)
    u = np.random.uniform(0, 1, sample_size)
    return f(u)

# todo: make instance of scipy.stats.rv_continuous
# todo: save y in log-space
# todo: enable saving in log-x
# todo: implement for discrete x (pmf)
# todo: point_estimate(), quantile(probs)
# todo: P.find_approximation(), which outputs a belief distribution p(model|data)
# todo: P.approx_normal(), P.approx_beta(), P.approx_lognormal(), ... (find best hyperparameters automatically)
class P:
    smoothing = 0.1

    def __init__(self, x, y=None):
        import scipy

        if isinstance(x, scipy.stats._distn_infrastructure.rv_frozen):
            if x.dist.__module__ == 'scipy.stats._continuous_distns':
                x = x.rvs(int(1e6))
                y = None # ignore argument
            else:
                raise ValueError("Only continuous functions supported!")
        if y is None:
            x, y = density(x, smoothing=P.smoothing)
        else:
            x, y = smooth(x, y, smoothing=P.smoothing)
            x, y = P._normalize(x,y)

        x, y = np.array(x), np.array(y)
        # pdf
        self.pdf = scipy.interpolate.interp1d(x, y, bounds_error=False, fill_value=0)
        # cdf
        dx = np.diff(x)
        y_centers = moving_avg(y,2)
        y_cs = np.cumsum(y_centers*dx)
        x_cs = moving_avg(x,2)
        self.cdf = scipy.interpolate.interp1d(x_cs, y_cs, bounds_error=False, fill_value=0)

    @property
    def x(self):
        return self.pdf.x

    @property
    def y(self):
        return self.pdf.y

    def __op__(self, other, op):
        x = np.concatenate([self.x,other.x])
        x.sort()
        y = op(x, self.pdf, other.pdf)
        return P(x,y) # normalizes in constructor

    def __add__(self, other):
        return self.__op__(other, lambda x, f, g: f(x)+g(x))

    def __sub__(self, other):
        return self.__op__(other, lambda x, f, g: f(x)-g(x))

    def __mul__(self, other):
        return self.__op__(other, lambda x, f, g: f(x)*g(x))

    def __truediv__(self, other):
        return self.__op__(other, lambda x, f, g: f(x)/g(x))

    def __call__(self, x):
        return self.pdf(x)

    def plot(self, *pltargs, **pltkwargs):
        plt.plot(self.x, self.y, *pltargs, **pltkwargs)

    def sample(self, size=1):
        u = np.random.uniform(0, 1, sample_size)
        return self.cdf(u)

    @property
    def nbytes(self):
        n_pdf = self.pdf.x.nbytes
        n_cdf = self.cdf.x.nbytes
        return n_pdf*2 + n_cdf*2

    @staticmethod # e.g. b = P.use(lambda p: binom.pmf(44, 274, p))
    def use(f, start=0, stop=1, size=int(1e4)):
        x = np.linspace(start, stop, size)
        y = [f(i) for i in x]
        return P(x,y) # normalizes in constructor

    @staticmethod
    def _normalize(x, y):
        dx = np.diff(x)
        y_centers = moving_avg(y,2)
        integral = np.sum(dx*y_centers)
        if integral == 0 or integral > 1e20 or np.isnan(integral):
            raise ValueError("Not normalizable!")
        y /= integral
        return x, y
