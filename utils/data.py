import numpy as np
import pandas as pd

# make a list out of a pd.corr() matrix
def corrList(corr, index_names=["feature 1", "feature 2"]):
    corr = corr.where(np.triu(np.ones(corr.shape), 1).astype(bool))
    corr = pd.DataFrame(corr.stack(), columns=["correlation"])
    corr.index.names = index_names
    return corr

def plot_dendrogram(X, method="ward", truncate_after=25, metric='euclidean', ax=None):
    from scipy.cluster.hierarchy import dendrogram, linkage
    Z = linkage(X, metric=metric, method=method)
    dendrogram(Z, truncate_mode='lastp', p=truncate_after, leaf_rotation=90, ax=ax)

def plot_dendrograms(X, methods=["single", "complete", "average", "centroid", "ward"], truncate_after=50, metric='euclidean'):
    from scipy.cluster.hierarchy import dendrogram, linkage

    fig, axes = plt.subplots(len(methods), figsize=(12,6*len(methods)))
    for ax, method in zip(axes, methods):
        ax.set_title(method)
        plot_dendrogram(X, method, truncate_after, metric, ax)

# from scipy.stats import circmean
def circular_mean(X, mod=360):
    rads = X*2*np.pi/mod
    av_sin = np.mean(np.sin(rads))
    av_cos = np.mean(np.cos(rads))
    av_rads = np.arctan2(av_sin,av_cos) % (2*np.pi) # move the negative half to the other side -> [0;2pi]
    return av_rads * mod/(2*np.pi)

# todo: data whitening, remove nan, intelligent data cleaning, automatic outlier detection