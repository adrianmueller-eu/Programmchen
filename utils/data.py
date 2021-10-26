import numpy as np
import pandas as pd

# make a list out of a pd.corr() matrix
def corrList(corr, index_names=["feature 1", "feature 2"]):
    corr = corr.where(np.triu(np.ones(corr.shape), 1).astype(bool))
    corr = pd.DataFrame(corr.stack(), columns=["correlation"])
    corr.index.names = index_names
    return corr

def plot_dendrogram(X, method="ward", truncate_after=25, metric='euclidean', ax=None):
    Z = linkage(X, metric=metric, method=method)
    dendrogram(Z, truncate_mode='lastp', p=truncate_after, leaf_rotation=90, ax=ax)

def plot_dendrograms(X, methods=["single", "complete", "average", "centroid", "ward"], truncate_after=50, metric='euclidean'):
    from scipy.cluster.hierarchy import dendrogram, linkage

    fig, axes = plt.subplots(len(methods), figsize=(12,6*len(methods)))

    for ax, method in zip(axes, methods):
        ax.set_title(method)
        plot_dendrogram(X, method, truncate_after, metric, ax)

# todo: data whitening, remove nan, intelligent data cleaning, automatic outlier detection
