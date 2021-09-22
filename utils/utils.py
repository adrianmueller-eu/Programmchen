import numpy as np

##############################
### also in 00.py
##############################

def moving_avg(x, w=3):
    return np.convolve(x, np.ones(w), 'valid') / w

def bins(data):
    return int(np.ceil(np.sqrt(len(data))))

def r(x, precision=5):
    return round(x, precision)
