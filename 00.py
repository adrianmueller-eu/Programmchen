import sys, os, time # For timing: %time foo()
import inspect
def getsource(f):
    print(inspect.getsource(f))
eps = sys.float_info.epsilon
### math
import numpy as np

try:
    integral # see if sage is loaded
except:
    from math import *
    from numpy import product as mul
    try:
       import sympy
    except:
       pass
try:
    from numpy.linalg import *
    from numpy.linalg import matrix_rank as rank
    from numpy.random import * # https://numpy.org/doc/stable/reference/random/generator.html

    import scipy
    from scipy.stats import * # https://docs.scipy.org/doc/scipy/reference/stats.html
    from scipy.optimize import minimize
    from scipy.linalg import *
except:
    pass

import itertools # https://docs.python.org/3/library/itertools.html
### visual
# import pandas as pd
# %matplotlib
import matplotlib.pyplot as plt # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.html#module-matplotlib.pyplot
try:
  from tqdm.autonotebook import tqdm as tq
except:
  pass
### options
# pd.set_option('display.max_rows', 500)
# pd.set_option('display.max_colwidth', None) # show complete text in df cells
# pd.set_option("display.precision", 3)
# np.set_printoptions(precision=3, suppress=True) # suppress == no scientific notation
# plt.style.use('ggplot') # nicer plots?
if "COLUMNS" in os.environ:
    cols = int(os.environ["COLUMNS"])
    if cols != 0:
        np.set_printoptions(linewidth=cols)

def bins(data):
    return int(np.ceil(np.sqrt(len(data))))

### more
# %lsmagic
# import warnings; warnings.filterwarnings('ignore')

# from ipywidgets import interact, FloatSlider, IntText
# FloatSlider(min=0, max=1, step=0.1, value=1, continuous_update=False)
# https://ipywidgets.readthedocs.io/en/latest/examples/Widget%20List.html

# import networkx as nx
# import tensorflow as tf
# import torch.nn as nn
# from scipy.optimize import minimize

# import stan
# import nest_asyncio
# nest_asyncio.apply()

from utils import *
#from utils.quantum import *
