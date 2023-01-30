import matplotlib.pyplot as plt
import numpy as np
import collections
from .mathlib import is_complex, is_symmetric, normalize, closest_prime_factors_to_sqrt
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

    if type(x) != np.ndarray:
        x = np.array(list(x))
    if y is not None and type(y) != np.ndarray:
        y = np.array(list(y))
    # plot
    if len(plt.get_fignums()) == 0:
        plt.figure(figsize=figsize)
    if fmt == ".":
        if y is None:
            y = x
            x = np.linspace(1,len(x),len(x))
        if is_complex(y):
            plt.scatter(x, y.real, label="real", **pltargs)
            plt.scatter(x, y.imag, label="imag", **pltargs)
            plt.legend()
        else:
            plt.scatter(x, y, **pltargs)
    elif y is not None:
        if is_complex(y):
            plt.plot(x, y.real, fmt, label="real", **pltargs)
            plt.plot(x, y.imag, fmt, label="imag", **pltargs)
            plt.legend()
        else:
            plt.plot(x, y, fmt, **pltargs)
    else:
        if is_complex(x):
            plt.plot(x.real, fmt, label="real", **pltargs)
            plt.plot(x.imag, fmt, label="imag", **pltargs)
            plt.legend()
        else:
            plt.plot(x, fmt, **pltargs)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if "label" in pltargs:
        plt.legend()
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)

# # basics, no log
# def hist(data, bins=None, xlabel="", title="", density=False):
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

def histogram(data, bins=None, xlog=False, density=False):
    if xlog:
        if not isinstance(bins, collections.Sequence):
            bins = logbins(data, num=bins)
    elif not bins:
        bins = bins_sqrt(data)
    return np.histogram(data, bins=bins, density=density)

def hist(data, bins=None, xlabel="", title="", xlog=False, ylog=False, density=False, colored=None, cmap="viridis", save_file=None):
    if type(bins) == str:
        if bins == "log":
            xlog = True
        elif bins == "loglog":
            xlog = True
            ylog = True
        else:
            xlabel = bins
        bins = None

    # create figure
    if colored:
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10,5), sharex=True, gridspec_kw={"height_ratios": [10, 1]})
        ax0 = ax[0]
    else:
        fig, ax0 = plt.subplots(figsize=(10,5))

    n, bins = histogram(data, bins=bins, xlog=xlog, density=density)
    ax0.hist(bins[:-1], bins, weights=n) # TODO: moving_avg(bins,2) instead of bins[:-1]?
    if xlog:
        ax0.set_xscale("log")
    if ylog:
        ax0.set_yscale("log")

    # visuals
    ax0.set_title(title)
    ax0.set_ylabel("density" if density else "frequency")
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

def scatter1d(data, xticks=None, alpha=.5, s=500, marker="|", **pltargs):
    fig = plt.figure(figsize=(10,1))
    ax = fig.gca()
    size = np.array(data).flatten().shape
    plt.scatter(data, np.zeros(*size), alpha=alpha, marker=marker, s=s, **pltargs)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_yticks([])
    if xticks:
        ax.set_xticks(xticks)
    fig.tight_layout()
    plt.show()

def colorize_complex(z):
    from colorsys import hls_to_rgb

    r = np.abs(z)
    a = np.angle(z)

    h = a / (2*np.pi)
    l = 1.0 - 1.0/(1.0 + r**0.3)
    s = 0.8

    c = np.vectorize(hls_to_rgb)(h,l,s)
    c = np.array(c).transpose(1,2,0) # convert shape (3,n,m) -> (n,m,3)
    return c

def imshow(a, figsize=(8,6), title="", cmap="hot", yticks=None, xticks=None, **pltargs):
    a = np.array(a)
    if np.prod(a.shape) == np.max(a.shape):
        a = a.flatten()
    fig = plt.figure(figsize=figsize)
    if len(a.shape) == 1:
        if a.shape[0] >= 100:
            # magic reshape
            best_divisor = np.prod(closest_prime_factors_to_sqrt(a.shape[0]))
            a = a.reshape(best_divisor, -1)
        else:
            a = a[:,None] # vertical
        if is_complex(a):
            img = colorize_complex(a)
            plt.imshow(img, aspect=5/a.shape[0], **pltargs)
        else:
            a = a.real
            img = plt.imshow(a, cmap=cmap, **pltargs)
            fig.colorbar(img, fraction=0.1, pad=0.05)
    elif len(a.shape) == 2:
        if is_complex(a):
            img = colorize_complex(a)
            plt.imshow(img, **pltargs)
        else:
            a = a.real
            img = plt.imshow(a, cmap=cmap, **pltargs)
            fig.colorbar(img, fraction=0.1, pad=0.05, shrink=0.87)
    else:
        raise ValueError(f"Array must be 2D or 1D, but shape was {a.shape}")

    if xticks is not None:
        plt.xticks(range(len(a)), xticks)
    if yticks is not None:
        plt.yticks(range(len(a)), yticks)

    plt.title(title)
    plt.show()

def complex_colorbar(figsize=(2,2)):
    imag, real = np.mgrid[-1:1:0.01,-1:1:0.01]
    imag = imag[::-1] # convention: turn counter-clockwise
    x = real + 1j*imag
    c = colorize_complex(x)
    plt.figure(figsize=figsize)
    plt.imshow(c)
    plt.xticks(np.linspace(0,200,5), np.linspace(-1,1,5))
    plt.yticks(np.linspace(0,200,5), np.linspace(-1,1,5)[::-1])
    plt.xlabel("real")
    plt.ylabel("imag")

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
        return 'white' if perceived_brightness(r,g,b) < 0.5 else 'black'

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
