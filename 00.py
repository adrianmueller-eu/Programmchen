import sys, os, time # For timing: %time foo()
from math import *
import numpy as np
from numpy.random import * # https://numpy.org/doc/stable/reference/random/generator.html
# import pandas as pd
import itertools # https://docs.python.org/3/library/itertools.html
from tqdm import tqdm

# %matplotlib
import matplotlib.pyplot as plt # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.html#module-matplotlib.pyplot

def bins(data):
    return int(np.ceil(np.sqrt(len(data))))

def r(x, precision=5):
    return round(x, precision)

# from ipywidgets import interact, FloatSlider, IntText
# FloatSlider(min=0, max=1, step=0.1, value=1, continuous_update=False)
# https://ipywidgets.readthedocs.io/en/latest/examples/Widget List.html

# import networkx as nx
# import tensorflow as tf
# import torch.nn as nn

# import stan
# import nest_asyncio
# nest_asyncio.apply()

def imshow(a, cmap_for_real="hot"):
    from colorsys import hls_to_rgb

    def colorize(z):
        r = np.abs(z)
        arg = np.angle(z)

        h = (arg + pi)  / (2 * np.pi) + 0.5
        l = 1.0 - 1.0/(1.0 + r**0.3)
        s = 0.8

        c = np.vectorize(hls_to_rgb) (h,l,s) # --> tuple
        c = np.array(c)  # -->  array of (3,n,m) shape, but need (n,m,3)
        c = c.transpose(1,2,0)
        return c

    def iscomplex(a):
        return np.iscomplex(a).any()
#         return a.dtype == "complex128"

    a = np.array(a)
    if len(a.shape) != 2:
        raise ValueError(f"Array must be 2D, but shape was {a.shape}")

    if iscomplex(a):
        img = colorize(a)
        plt.imshow(img)
    else:
        a = a.real
        plt.imshow(a, cmap=cmap_for_real)
        plt.colorbar()
    plt.show()

def hist(data, title="", xlabel="", colored=None, cmap="viridis", save_file=None, bins=None):
    if colored:
       fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10,5), sharex=True, gridspec_kw={"height_ratios": [10, 1]})
       ax0 = ax[0]
    else:
       fig, ax0 = plt.subplots(figsize=(10,5))

    ax0.set_title(title)
    if bins is None:
        bins = int(np.ceil(np.sqrt(len(data))))
    ax0.hist(data, bins=bins)
    ax0.set_ylabel("Frequency")
    ax0.spines["top"].set_visible(False)
    ax0.spines["right"].set_visible(False)
    ax0.spines["bottom"].set_visible(False)

    if colored:
        ax[1].scatter(data, np.zeros(*data.shape), alpha=.5, c=colored, cmap=cmap, marker="|", s=500)
        # ax[1].axis("off")
        ax[1].set_xlabel(xlabel)
        ax[1].set_yticks([])
        ax[1].spines["top"].set_visible(False)
        ax[1].spines["right"].set_visible(False)
        ax[1].spines["left"].set_visible(False)

        norm = plt.Normalize(vmin=min(colored), vmax=max(colored))
        sm = plt.cm.ScalarMappable(norm=norm, cmap=plt.get_cmap(cmap))
        cb = plt.colorbar(sm, ax=ax, fraction=0.05, pad=0.01, aspect=50)

    plt.show()

    if save_file:
        plt.savefig(save_file)


def moving_avg(x, w=3):
    return np.convolve(x, np.ones(w), 'valid') / w

# converts 1-d data into a pdf, smoothing in [0,1]
def density(r, smoothing=0.1, plot=True, label=None):
    bins_c = bins(r)
    n, bin_edges = np.histogram(r, bins_c, density=True)
    bin_centers = moving_avg(bin_edges,2)
    if not smoothing:
        return bin_centers, n

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

    window = max(2,int(smoothing*bins_c))
    x, y = bin_centers, savitzky_golay(n, window, min(window-2,3))
    # normalization
    dx = x[1] - x[0]
    y /= np.sum(y*dx)

    if plot:
        plt.plot(x, y, label=label)
        top = max(plt.gca().get_ylim()[1], 1.05*np.max(y))
        plt.ylim(bottom=0, top=top)
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
