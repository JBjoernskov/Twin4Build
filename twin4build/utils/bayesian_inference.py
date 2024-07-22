import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde

def generate_quantiles(x, p=np.array([0.25, 0.5, 0.75])):
    '''
    Calculate empirical quantiles.

    Args:
        * **x** (:class:`~numpy.ndarray`): Observations from which to generate quantile.
        * **p** (:class:`~numpy.ndarray`): Quantile limits.

    Returns:
        * (:class:`~numpy.ndarray`): Interpolated quantiles.
    '''
    
    # extract number of rows/cols from np.array
    n = x.shape[0]
    if n==1:
        return x
    # define vector valued interpolation function
    xpoints = np.arange(0, n, 1)
    interpfun = interp1d(xpoints, np.sort(x, 0), axis=0)
    # evaluation points
    itpoints = (n - 1)*p
    return interpfun(itpoints)

def generate_mean(x):
    '''
    Calculate empirical mode.

    Args:
        * **x** (:class:`~numpy.ndarray`): Observations from which to generate mode.
        * **p** (:class:`~numpy.ndarray`): Number of bins.

    Returns:
        * (:class:`~numpy.ndarray`): Mode from histogram.
    '''
    means = np.mean(x, axis=0)
    return means

def generate_mode(x, n_bins=50):
    '''
    Calculate empirical mode.

    Args:
        * **x** (:class:`~numpy.ndarray`): Observations from which to generate mode.
        * **p** (:class:`~numpy.ndarray`): Number of bins.

    Returns:
        * (:class:`~numpy.ndarray`): Mode from histogram.
    '''
    ###
    # n_timesteps = x.shape[1]
    # hist = [np.histogram(x[:,i], bins=n_bins) for i in range(n_timesteps)]
    # frequency = np.array([el[0] for el in hist])
    # edges = np.array([el[1] for el in hist])
    # mode_indices = np.argmax(frequency,axis=1)
    # modes = edges[np.arange(n_timesteps), mode_indices]
    ###
    modes = np.zeros((x.shape[1]))
    for t in range(x.shape[1]):
        x_t = x[:,t]
        xpoints = np.linspace(np.min(x_t), np.max(x_t), 300)
        kde = gaussian_kde(x_t)
        p = kde.pdf(xpoints)
        modes[t] = xpoints[p.argmax()]
    return modes