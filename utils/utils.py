import numpy as np

def moving_avg(x, w=3):
    return np.convolve(x, np.ones(w), 'valid') / w

def r(x, precision=5):
    return round(x, precision)

def bins_sqrt(data):
    return int(np.ceil(np.sqrt(len(data))))

def logbins(data, start=None, stop=None, num=None, scale=2):
    if start is None:
        start = min(data)/scale
    if start <= 0:
        data = np.array(data)
        data_pos = data[data > 0]
        print("Warning: Data set contains non-positive numbers (%.2f%%). They will be excluded for the plot." % (100*(len(data_pos)/len(data))))
        data = data_pos
        start = min(data)/scale
    if stop is None:
        stop = max(data)*scale
    if num is None:
        num = bins_sqrt(data)
    return 10**(np.linspace(np.log10(start),np.log10(stop),num))
