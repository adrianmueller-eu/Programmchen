import matplotlib.pyplot as plt
import numpy as np
from math import factorial
from .utils import *

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

        window = max(2,int(smoothing*len(bin_centers)))
        x, y = bin_centers, savitzky_golay(n, window, min(window-2,3))
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
    f = interp1d(y_cs, x)
    u = np.random.uniform(min(y_cs), max(y_cs), sample_size)
    return f(u)
