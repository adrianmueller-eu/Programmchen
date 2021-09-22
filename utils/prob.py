import matplotlib.pyplot as plt
import numpy as np

# converts 1-d data into a pdf, smoothing in [0,1]
def density(data, plot=True, label=None, smoothing=0.1):
    bins_c = int(np.ceil(np.sqrt(len(data))))
    n, bin_edges = np.histogram(data, bins_c, density=True)
    bin_centers = np.convolve(bin_edges, np.ones(2), 'valid')
    if smoothing:
        # https://scipy.github.io/old-wiki/pages/Cookbook/SavitzkyGolay
        def savitzky_golay(y, window_size, order, deriv=0, rate=1):
            if window_size < order + 2:
                raise TypeError("window_size is too small for the polynomials order")
            order_range = range(order+1)
            half_window = (window_size -1) // 2
            # precompute coefficients
            b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
            m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
            # pad the signal at the extremes with
            # values taken from the signal itself
            firstvals = y[0] - np.abs(y[1:half_window+1][::-1] - y[0])
            lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
            y = np.concatenate((firstvals, y, lastvals))
            return np.convolve(m[::-1], y, mode='valid')

        window = max(2,int(smoothing*bins_c))
        x, y = bin_centers, savitzky_golay(n, window, min(window-2,3))
        # normalization
        dx = x[1] - x[0]
        y /= np.sum(y*dx)
    else:
        x,y = bin_centers, n

    if plot:
        plt.plot(x, y, label=label)
        top = max(plt.gca().get_ylim()[1], 1.05*np.max(y))
        plt.ylim(bottom=0, top=top)
        if label:
            plt.legend()

    return x, y

# x,y should be an output of "density" above
def resample(x, y, sample_size=int(1e6)):
    dx = x[1] - x[0]
    y_cs = np.cumsum(y*dx)
    from scipy.interpolate import interp1d
    f = interp1d(y_cs, x)
    u = np.random.uniform(min(y_cs), max(y_cs), sample_size)
    return f(u)