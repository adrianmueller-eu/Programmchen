import matplotlib.pyplot as plt
import numpy as np
import collections
from .utils import *

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

def hist(data, title="", xlabel="", colored=None, cmap="viridis", save_file=None, bins=None, log=False, density=False):
    # create figure
    if colored:
       fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10,5), sharex=True, gridspec_kw={"height_ratios": [10, 1]})
       ax0 = ax[0]
    else:
       fig, ax0 = plt.subplots(figsize=(10,5))

    # bins
    if log:
       if not isinstance(bins, collections.Sequence):
            bins = logbins(data, num=bins)
       ax0.set_xscale("log")
    else:
       bins = bins_sqrt(data)
    n, bins, _ = ax0.hist(data, bins=bins, density=density)

    # visuals
    ax0.set_title(title)
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

    return n, bins

def scatter1d(data, xticks=None, **pltargs):
    fig = plt.figure(figsize=(10,1))
    ax = fig.gca()
    plt.scatter(data, np.zeros(*data.shape), alpha=.5, marker="|", s=500, *pltargs)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_yticks([])
    if xticks:
        ax.set_xticks(xticks)
    fig.tight_layout()
    plt.show()

# coloring for pd.corr() matrix
def corrColor(corr, threshold=0.6):
    def highlight(value):
        if np.isnan(value):
            bg_color = 'white'
            color = 'white'
        elif value < -threshold:
            bg_color = 'red'
            color = 'white'
        elif value > threshold:
            bg_color = 'green'
            color = 'white'
        else:
            bg_color = 'white'
            color = 'black'
        return f"background-color: %s; color: %s" % (bg_color, color)

    corr = corr.where(np.tril(np.ones(corr.shape), -1).astype(bool))
    return corr.style.applymap(highlight)

# make a list out of a pd.corr() matrix
def corrList(corr):
    corr = corr.where(np.triu(np.ones(corr.shape), 1).astype(bool))
    corr = pd.DataFrame(corr.stack(), columns=["correlation"])
    corr.index.names = ["feature 1", "feature 2"]
    return corr