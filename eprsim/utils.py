import glob
import os


import numpy
import scipy
from scipy import interpolate

# Random number generator
rng = numpy.random.default_rng()


def inverse_transform_sampling(data, n_bins=40, n_samples=1000):
    hist, bin_edges = numpy.histogram(data, bins=n_bins, density=True)
    cum_values = numpy.zeros(bin_edges.shape)
    cum_values[1:] = numpy.cumsum(hist * numpy.diff(bin_edges))
    inv_cdf = interpolate.interp1d(cum_values, bin_edges)
    r = rng.rand(n_samples)
    return inv_cdf(r)


def pdf_sampler(data, bins=100):
    hist, edges = numpy.histogram(data, bins=bins, density=True)
    cum = numpy.zeros_like(hist)
    cum = numpy.cumsum(hist*numpy.diff(edges))
    return scipy.stats.rv_histogram((cum, edges))


def sample_pdf(func, low=0, high=1, size=None):
    edges = numpy.linspace(low, high, 1000000)
    x = edges[:-1] + numpy.diff(edges) / 2
    y = (func(x) * 1e6).astype(int)

    dist = scipy.stats.rv_histogram((y * 1e6))
    return dist.rvs(size=size)


def rand_unit_vec(size=None, theta=None):
    """
    Generate a random unit vector uniformly distributed on a sphere
    """
    theta = numpy.random.uniform(0, numpy.pi * 2, size=size) if theta is None else numpy.radians(theta)
    u = numpy.random.uniform(-1, 1, size=size)
    v = numpy.sqrt(1 - u ** 2)
    if size:
        return numpy.array([v * numpy.cos(theta), v * numpy.sin(theta), u]).T
    else:
        return numpy.array([v * numpy.cos(theta), v * numpy.sin(theta), u])


def rand_plane_vec(size=None, theta=None):
    """
    Generate a random unit vector uniformly distributed on a circle
    """
    theta = numpy.random.uniform(0, numpy.pi * 2, size=size) if theta is None else numpy.radians(theta)
    u = 0
    v = numpy.sqrt(1 - u ** 2)
    if size:
        return numpy.array([v * numpy.cos(theta), v * numpy.sin(theta), numpy.array([u] * size)]).T
    else:
        return numpy.array([v * numpy.cos(theta), v * numpy.sin(theta), u])


def get_latest(*patterns):
    """
    Get the latest file name in current working directory matching each of the glob patterns provided
    :param patterns: file glob pattern
    :return: one file for each pattern or None if no file is found
    """

    file_groups = [glob.glob(pattern) for pattern in patterns]
    for group in file_groups:
        group.sort(key=lambda x: -os.path.getmtime(x))
    return [None if not group else group[0] for group in file_groups]


def match(a, b):
    """
    Find and return indices in array a and b that produce the pairs with values closest to each other.
    entries not matched are removed

    :param a: array of numbers
    :param b: array of numbers
    :return: tuple of indices into a and b that generate matched arrays of the same length
    """

    # find unique positions in array "a", which to place "b" values and their corresponding b positions
    afi1 = numpy.searchsorted(a, b)
    bi1 = numpy.nonzero(numpy.diff(afi1))[0]
    ai1 = afi1[bi1]

    # find unique positions in b in which to place a values and their corresponding a positions
    bfi2 = numpy.searchsorted(b, a)
    ai2 = numpy.nonzero(numpy.diff(bfi2))[0]
    bi2 = bfi2[ai2]

    ins = numpy.searchsorted(ai1, ai2)
    ai = numpy.insert(ai1, ins, ai2)
    bi = numpy.insert(bi1, ins, bi2)

    # ai may no longer be unique sets but are still sorted. At most duplicates
    # check non-unique pairs of positions and keep the one with the minimum difference
    ai_nu = numpy.nonzero(numpy.diff(ai) <= 0)[0]
    bi_nu = numpy.nonzero(numpy.diff(bi) <= 0)[0]
    while bi_nu.shape[0] or bi_nu.shape[0]:
        valid = numpy.ones_like(ai, dtype=bool)
        for idx in numpy.concatenate((ai_nu, bi_nu)):
            i = idx
            j = idx + 1
            remove = max(
                (abs(a[ai[i]] - b[bi[i]]), i),
                (abs(a[ai[j]] - b[bi[j]]), j),
            )
            valid[remove[1]] = False
        ai = ai[valid]
        bi = bi[valid]
        ai_nu = numpy.nonzero(numpy.diff(ai) <= 0)[0]
        bi_nu = numpy.nonzero(numpy.diff(bi) <= 0)[0]

    return ai, bi