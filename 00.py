import sys, os, time # For timing: %time foo()
### math
import numpy as np
try:
    integral # see if sage is loaded
except:
    from math import *
try:
    import scipy
    from scipy.stats import *
except:
    from numpy.random import * # https://numpy.org/doc/stable/reference/random/generator.html
import itertools # https://docs.python.org/3/library/itertools.html
### visual
# import pandas as pd
# %matplotlib
import matplotlib.pyplot as plt # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.html#module-matplotlib.pyplot
from tqdm import tqdm
### options
# pd.set_option('display.max_rows', 500)
# np.set_printoptions(precision=3, suppress=True) # suppress == no scientific notation

def bins(data):
    return int(np.ceil(np.sqrt(len(data))))

### more
# from ipywidgets import interact, FloatSlider, IntText
# FloatSlider(min=0, max=1, step=0.1, value=1, continuous_update=False)
# https://ipywidgets.readthedocs.io/en/latest/examples/Widget List.html

# import networkx as nx
# import tensorflow as tf
# import torch.nn as nn
# from scipy.optimize import minimize

# import stan
# import nest_asyncio
# nest_asyncio.apply()

from utils import *
