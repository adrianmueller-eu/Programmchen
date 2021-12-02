import matplotlib.pyplot as plt
import numpy as np
import collections
from .utils import *

def plot(x,y=None, fmt="-", figsize=(10,8), xlabel="", ylabel="", title="", **pltargs):
    # make it a bit intelligent
    if type(x) == tuple and len(x) == 2:
        title  = ylabel
        ylabel = xlabel
        if type(figsize) == str:
            xlabel = figsize
            if type(fmt) == tuple:
                figsize= fmt
                if type(y) == str:
                    fmt = y
        y = x[1]
        x = x[0]
    elif type(y) == str: # skip y
        title  = ylabel
        ylabel = xlabel
        if type(figsize) == str:
            xlabel = figsize
            if type(fmt) == tuple:
                figsize= fmt
        fmt=y
        y=None
    elif type(y) == tuple and len(y) == 2 and type(y[0]) == int and type(y[1]) == int: # skip y and fmt
        title  = xlabel
        if type(figsize) == str:
            ylabel = figsize
            xlabel = fmt
        figsize= y
        fmt=None
        y=None
    if type(fmt) == tuple: # skip fmt
        title  = ylabel
        ylabel = xlabel
        if type(figsize) == str:
            xlabel = figsize
        figsize= fmt
        fmt=None
    elif type(figsize) == str: # skip figsize
        title  = ylabel
        ylabel = xlabel
        xlabel = figsize
        figsize= (10,8)

    if fmt is None:
        fmt = "-"
    # plot
    if len(plt.get_fignums()) == 0:
        plt.figure(figsize=figsize)
    if fmt == ".":
        if y is None:
            y = x
            x = np.linspace(1,len(x),len(x))
        plt.scatter(x, y, **pltargs)
    elif y is not None:
        plt.plot(x, y, fmt, **pltargs)
    else:
        plt.plot(x, fmt, **pltargs)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)

# # basics, no log
# def hist(data, xlabel="", title="", bins=None, density=False):
#     def bins_sqrt(data):
#         return int(np.ceil(np.sqrt(len(data))))
#
#     plt.figure(figsize=(10,5))
#
#     # bins
#     if not bins:
#         bins = bins_sqrt(data)
#     n, bins, _ = plt.hist(data, bins=bins, density=density)
#
#     # visuals
#     plt.title(title)
#     plt.ylabel("Density" if density else "Frequency")
#     plt.xlabel(xlabel)
#     plt.gca().spines["top"].set_visible(False)
#     plt.gca().spines["right"].set_visible(False)
#     plt.gca().spines["bottom"].set_visible(False)
#     return n, bins

# # basics
# def hist(data, xlabel="", title="", bins=None, log=False, density=False):
#     import collections
#
#     def bins_sqrt(data):
#         return int(np.ceil(np.sqrt(len(data))))
#
#     def logbins(data, start=None, stop=None, num=None, scale=2):
#         if start is None:
#             start = min(data)/scale
#         if stop is None:
#             stop = max(data)*scale
#         if num is None:
#             num = bins_sqrt(data)
#         return 10**(np.linspace(np.log10(start),np.log10(stop),num))
#
#     plt.figure(figsize=(10,5))
#
#     # bins
#     if log:
#         if not isinstance(bins, collections.Sequence):
#             bins = logbins(data, num=bins)
#         plt.xscale("log")
#     elif not bins:
#         bins = bins_sqrt(data)
#     n, bins, _ = plt.hist(data, bins=bins, density=density)
#
#     # visuals
#     plt.title(title)
#     plt.ylabel("Density" if density else "Frequency")
#     plt.xlabel(xlabel)
#     plt.gca().spines["top"].set_visible(False)
#     plt.gca().spines["right"].set_visible(False)
#     plt.gca().spines["bottom"].set_visible(False)
#     return n, bins

def histogram(data, bins=None, log=False, density=False):
    if log:
        if not isinstance(bins, collections.Sequence):
            bins = logbins(data, num=bins)
    elif not bins:
        bins = bins_sqrt(data)
    return np.histogram(data, bins=bins, density=density)

def hist(data, title="", xlabel="", colored=None, cmap="viridis", save_file=None, bins=None, log=False, density=False):
    # create figure
    if colored:
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10,5), sharex=True, gridspec_kw={"height_ratios": [10, 1]})
        ax0 = ax[0]
    else:
        fig, ax0 = plt.subplots(figsize=(10,5))

    n, bins = histogram(data, bins=bins, log=log, density=density)
    ax0.hist(bins[:-1], bins, weights=n) # TODO: moving_avg(bins,2) instead of bins[:-1]?
    if log:
        ax0.set_xscale("log")

    # visuals
    ax0.set_title(title)
    ax0.set_ylabel("Density" if density else "Frequency")
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
    else:
        ax0.set_xlabel(xlabel)

    plt.show()

    if save_file:
        plt.savefig(save_file)

    return n, bins

def scatter1d(data, xticks=None, **pltargs):
    fig = plt.figure(figsize=(10,1))
    ax = fig.gca()
    size = np.array(data).flatten().shape
    plt.scatter(data, np.zeros(*size), alpha=.5, marker="|", s=500, *pltargs)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_yticks([])
    if xticks:
        ax.set_xticks(xticks)
    fig.tight_layout()
    plt.show()

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

def bar(heights, log=False):
    N = len(heights)
    plt.figure(figsize=(int(np.ceil(N/2)),6))
    plt.bar(range(N), height=heights)
    plt.xticks(range(N))
    if log:
        plt.yscale("log")
    plt.tight_layout()
    plt.show()


def rgb(r,g=1.0,b=1.0,a=1.0, as255=False):
    conv = 1 if as255 else 1/255
    if type(r) == str:
        s = r.split("#")[-1]
        r = int(s[0:2],16)*conv
        g = int(s[2:4],16)*conv
        b = int(s[4:6],16)*conv
        if len(s) > 6:
            a = int(s[6:8],16)*conv
        else:
            a = 1
        return (r,g,b,a)
    return "#" + "".join('{0:02X}'.format(int(v/conv)) for v in [r,g,b,a])

def perceived_brightness(r,g,b,a=None): # a is ignored
    return 0.2126 * r + 0.7152 * g + 0.0722 * b

# coloring for pd.DateFrame
# todo: use package "webcolors" (e.g. name to rgb)
def pdcolor(df, threshold=None, minv=None, maxv=None, colors=['#ff0000', '#ffffff', '#069900'], tril_if_symmetric=True, bi=False):
    def blackorwhite(r,g=None,b=None):
        if g is None:
            r,g,b,a = rgb(r)
        return 'white' if perceived_brightness(r,b,g) < 0.42 else 'black'

    df = df.dropna(thresh=1).T.dropna(thresh=1).T # filter out rows and cols with no data
    if bi:
        colors = colors[1:]
    if threshold:
        def highlight(value):
            if np.isnan(value):
                bg_color = 'white'
                color = 'white'
            elif value < -threshold:
                bg_color = colors[0]
                color = blackorwhite(bg_color)
            elif value > threshold:
                bg_color = colors[-1]
                color = blackorwhite(bg_color)
            else:
                bg_color = 'white'
                color = 'black'
            return f"background-color: %s; color: %s" % (bg_color, color)
    else:
        if len(colors) < 2:
            raise ValueError("Please give at least two colors!")
        if minv is None and maxv is None and len(df.columns) == len(df.index) and (df.columns == df.index).all(): # corr matrix!
            maxv = 1
            minv = -1
        if not maxv:
            maxv = df.max().max()
        if not minv:
            minv = df.min().min()
        if maxv <= minv:
            raise ValueError(f"Maxv must be higher than minv, but was: %f <= %f" % (maxv, minv))

        def getRGB(v):
            scaled = (v - minv)/(maxv - minv) * (len(colors)-1)
            scaled = max(0,min(scaled, len(colors)-1-1e-10)) #[0;len(colors)-1[
            subarea = int(np.floor(scaled))
            low_c, high_c = colors[subarea], colors[subarea+1] # get frame
            low_c, high_c = np.array(rgb(low_c)), np.array(rgb(high_c)) # convert to (r,b,g,a)
            r,g,b,a = (scaled-subarea)*(high_c-low_c) + low_c
            return rgb(r,g,b,a), blackorwhite(r,g,b)

        def highlight(value):
            if np.isnan(value):
                bg_color = 'white'
                color = 'white'
            else:
                bg_color, color = getRGB(value)
            return f"background-color: %s; color: %s" % (bg_color, color)

    if tril_if_symmetric and is_symmetric(df):
        df = df.where(np.tril(np.ones(df.shape), -1).astype(bool))
        df = df.dropna(thresh=1).T.dropna(thresh=1).T
    return df.style.applymap(highlight)
