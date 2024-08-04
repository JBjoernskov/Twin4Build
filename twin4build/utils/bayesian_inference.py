import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
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

def next_pow_two(n):
    i = 1
    while i < n:
        i = i << 1
    return i

def autocorr_func_1d(x, norm=True):
    x = np.atleast_1d(x)
    if len(x.shape) != 1:
        raise ValueError("invalid dimensions for 1D autocorrelation function")
    n = next_pow_two(len(x))

    # Compute the FFT and then (from that) the auto-correlation function
    f = np.fft.fft(x - np.mean(x), n=2 * n)
    acf = np.fft.ifft(f * np.conjugate(f))[: len(x)].real
    acf /= 4 * n

    # Optionally normalize
    if norm:
        acf /= acf[0]

    return acf



# Automated windowing procedure following Sokal (1989)
def auto_window(taus, c):
    m = np.arange(len(taus)) < c * taus
    if np.any(m):
        return np.argmin(m)
    return len(taus) - 1

# Following the suggestion from Goodman & Weare (2010)
def autocorr_gw2010(y, c=5):
    y = y.transpose()
    f = autocorr_func_1d(np.mean(y, axis=0))
    taus = 2.0 * np.cumsum(f) - 1.0
    window = auto_window(taus, c)
    return taus[window]

def get_iac_corr_first(y, c=5.0):
    '''
    Calculate integrated autocorrelation time.
    '''
    y = y.transpose()
    f = np.zeros(y.shape[1])
    for yy in y:
        f += autocorr_func_1d(yy)
    f /= len(y)
    taus = 2.0 * np.cumsum(f) - 1.0
    window = auto_window(taus, c)
    return taus[window]

def get_iac_old(y, interval=10):
    """
    axis 0: number of samples
    axis 1: number of temperatures
    axis 2: number of walkers
    axis 3: number of parameters
    """
    nsamples = y.shape[0]
    ntemps = y.shape[1]
    npar = y.shape[3]
    idx = np.arange(1, nsamples, interval)
    iac = np.zeros((idx.shape[0], ntemps, npar))
    for i, idx_ in enumerate(idx):
        print(i)
        for j in range(ntemps):
            for k in range(npar):
                iac[i-1, j, k] = get_iac_corr_first(y[:idx_, j, :, k])
    return iac,idx


def get_iac(data, C, window=None):
    def _get_iac(data, C, window=None):
        """
        axis 0: number of samples
        axis 1: number of walkers
        """
        # initial processing
        nsamples = data.shape[0]
        nwalkers = data.shape[1]
        
        if window is None:
            window = nsamples

        acf_mean = 0
        for i in range(nwalkers):
            data_ = data[:, i]
            avg = np.average(data_)
            data_centered = data_ - avg
            # auto-covariance function
            autocov = np.empty(window)
            for j in range(window):
                autocov[j, ] = np.dot(data_centered[:nsamples - j], data_centered[j:])
            autocov /= nsamples

            # autocorrelation function
            acf = autocov / autocov[0]
            acf_mean += acf
        acf_mean /= nwalkers
        acf = acf_mean

        # integrate autocorrelation function
        j_max_v = np.arange(window)
        tau_int_v = np.zeros(window)
        for j_max in j_max_v:
            tau_int_v[j_max] = 0.5 + np.sum(acf[1:j_max + 1])

        # find j_max
        j_max = 0
        while j_max < C * tau_int_v[j_max]:
            j_max += 1

        # wrap it up
        tau_int = tau_int_v[j_max]
        N_eff = nsamples / (2 * tau_int)
        sem = np.sqrt(autocov[0] / N_eff)

        # # create ACF plot
        # fig = plt.figure(figsize=(10, 6))
        # plt.gca().axhline(0, color="gray",linewidth=1)
        # plt.plot(acf)
        # plt.xlabel("lag time $j$")
        # plt.ylabel(r"$\hat{K}^{XX}_j$")
        # plt.show()

        # # create integrated ACF plot
        # fig = plt.figure(figsize=(10, 6))
        # plt.plot(j_max_v[1:], C * tau_int_v[1:])
        # plt.ylim(plt.gca().get_ylim()) # explicitly keep the limits of the first plot
        # plt.plot(j_max_v[1:], j_max_v[1:])
        # plt.plot([j_max], [C * tau_int_v[j_max]], "ro")
        # plt.xscale("log")
        # plt.xlabel(r"sum length $j_\mathrm{max}$")
        # plt.ylabel(r"$C \times \hat{\tau}_{X, \mathrm{int}}$")
        # plt.title("")
        # plt.show()

        # # print out stuff
        # print(f"Mean value: {avg:.4f}")
        # print(f"Standard error of the mean: {sem:.4f}")
        # print(f"Integrated autocorrelation time: {tau_int:.2f} time steps")
        # print(f"Effective number of samples: {N_eff:.1f}")

        return tau_int
    
    nsamples = data.shape[0]
    ntemps = data.shape[1]
    nwalkers = data.shape[2]
    npar = data.shape[3]
    iac = np.zeros((ntemps, npar))
    for i in range(ntemps):
        for j in range(npar):
            iac[i, j] = _get_iac(data[:, i, :, j], C, window)
    

        # time_series_1 = np.mean(y[:,0,:,0], axis=1)

        # # Numpy solution
        # time_series_1_centered = time_series_1 - np.average(time_series_1)
        # autocov = np.empty(nsamples)

        # for j in range(nsamples):
        #     autocov[j] = np.dot(time_series_1_centered[:nsamples - j], time_series_1_centered[j:])
        # autocov /= nsamples

        
        # fig = plt.figure(figsize=(10, 6))
        # fig.suptitle("Autocovariance function")
        # plt.gca().axhline(0, color="gray", linewidth=1)
        # plt.plot(autocov)
        # plt.xlabel("lag time $j$")
        # plt.ylabel(r"$\hat{K}^{XX}_j$")
        

        # # compute the ACF
        # acf = autocov / autocov[0]

        # fig = plt.figure(figsize=(10, 6))
        # fig.suptitle("Autocorrelation function")
        # plt.gca().axhline(0, color="gray", linewidth=1)
        # plt.plot(acf)
        # plt.xlabel("lag time $j$")
        # plt.ylabel(r"$\hat{K}^{XX}_j$")
        


        # # integrate the ACF (suffix _v for vectors)
        # j_max_v = np.arange(nsamples)
        # tau_int_v = np.zeros(nsamples)
        # for j_max in j_max_v:
        #     tau_int_v[j_max] = 0.5 + np.sum(acf[1:j_max + 1])

        # # plot
        # fig = plt.figure(figsize=(10, 6))
        # fig.suptitle("Integrated autocorrelation time")
        # plt.plot(j_max_v[1:], tau_int_v[1:], label="numerical summing")
        # plt.xscale("log")
        # plt.xlabel(r"sum length $j_\mathrm{max}$")
        # plt.ylabel(r"$\hat{\tau}_{X, \mathrm{int}}$")
        # plt.legend()
        


        # C = 5.0

        # # determine j_max
        # j_max = 0
        # while j_max < C * tau_int_v[j_max]:
        #     j_max += 1


        # # plot
        # fig = plt.figure(figsize=(10, 6))
        # fig.suptitle("Find j")
        # plt.plot(j_max_v[1:], C * tau_int_v[1:])
        # plt.plot(j_max_v[1:], j_max_v[1:])
        # plt.plot([j_max], [C * tau_int_v[j_max]], "ro")
        # plt.xscale("log")
        # # plt.ylim((0, 50))
        # plt.xlabel(r"sum length $j_\mathrm{max}$")
        # plt.ylabel(r"$C \times \hat{\tau}_{X, \mathrm{int}}$")
        

        # print(f"j_max = {j_max}")


        # tau_int = tau_int_v[j_max]
        # print(f"Integrated autocorrelation time: {tau_int:.2f} time steps\n")

        # N_eff = nsamples / (2 * tau_int)
        # print(f"Original number of samples: {nsamples}")
        # print(f"Effective number of samples: {N_eff:.1f}")
        # print(f"Ratio: {N_eff / nsamples:.3f}\n")

        # sem = np.sqrt(autocov[0] / N_eff)
        # print(f"Standard error of the mean: {sem:.4f}")

        # plt.show()

    return iac










