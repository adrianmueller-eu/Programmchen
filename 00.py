import sys, os, time # For timing: %time foo()
from math import *
import numpy as np
try:
    import scipy
    from scipy.stats import *
except:
    from numpy.random import * # https://numpy.org/doc/stable/reference/random/generator.html
# import pandas as pd
import itertools # https://docs.python.org/3/library/itertools.html
from tqdm import tqdm

# %matplotlib
import matplotlib.pyplot as plt # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.html#module-matplotlib.pyplot

def bins(data):
    return int(np.ceil(np.sqrt(len(data))))

# from ipywidgets import interact, FloatSlider, IntText
# FloatSlider(min=0, max=1, step=0.1, value=1, continuous_update=False)
# https://ipywidgets.readthedocs.io/en/latest/examples/Widget List.html

# import networkx as nx
# import tensorflow as tf
# import torch.nn as nn

# import stan
# import nest_asyncio
# nest_asyncio.apply()

from utils import *
