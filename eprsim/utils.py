import scipy
from scipy import interpolate
import numpy

# Random number generator
rng = numpy.random.default_rng()


def inverse_transform_sampling(data, n_bins=40, n_samples=1000):
    hist, bin_edges = numpy.histogram(data, bins=n_bins, density=True)
    cum_values = numpy.zeros(bin_edges.shape)
    cum_values[1:] = numpy.cumsum(hist * numpy.diff(bin_edges))
    inv_cdf = interpolate.interp1d(cum_values, bin_edges)
    r = rng.rand(n_samples)
    return inv_cdf(r)


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
    theta = numpy.random.uniform(0, numpy.pi*2, size=size) if theta is None else numpy.radians(theta)
    u = numpy.random.uniform(-1, 1, size=size)
    v = numpy.sqrt(1-u**2)
    if size:
        return numpy.array([v*numpy.cos(theta), v*numpy.sin(theta), u]).T
    else:
        return numpy.array([v*numpy.cos(theta), v*numpy.sin(theta), u])


def rand_plane_vec(size=None, theta=None):
    """
    Generate a random unit vector uniformly distributed on a circle
    """
    theta = numpy.random.uniform(0, numpy.pi*2, size=size) if theta is None else numpy.radians(theta)
    u = 0
    v = numpy.sqrt(1-u**2)
    if size:
        return numpy.array([v*numpy.cos(theta), v*numpy.sin(theta), numpy.array([u]*size)]).T
    else:
        return numpy.array([v*numpy.cos(theta), v*numpy.sin(theta), u])


def match(a, b):
    """
    Find and return indices in array a and b that produce the pairs with values closest to each other.
    entries not matched are removed
    """

    # find unique positions in a in which to place b values and their corresponding b positions
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


def jal_match(a_times, b_times):
    """
    Coincidence detection algorithm adapted from Jan Ake Larsson's version
    in BellTiming.

    http://people.isy.liu.se/jalar/belltiming/
    <jalar@mai.liu.se>
    This function is copyrighted to Jan Ake Larsson and Licensed under the GPL
    See the License file at the above website for details on re-use
    """
    np = numpy
    # Index is the index of Alice's data, everywhere but in b_times.
    #
    # Find the immediately following detections at Bob's site.
    # Only do \.../
    # If needed I'll add /... and ...\ starts and ends later.
    # An entry == len(b_times) here means no detection follows
    following = np.searchsorted(b_times, a_times)
    # Uniqify: At most one makes a pair with another
    i = np.nonzero(following[:-1] != following[1:])[0]
    ai_f = i
    bi_f = following[ai_f]
    ai_p = i + 1
    bi_p = following[ai_p] - 1
    # At this point, bi_f contains Bob's index of the detection
    # following the ones that have the indices ai_f at Alice.
    #
    # Time difference calculation
    diff_f = b_times[bi_f] - a_times[ai_f]
    diff_p = a_times[ai_p] - b_times[bi_p]
    # Link notation below: \ a detection at Bob follows a detection at
    # Alice, / a detection at Bob precedes a detection at Alice.
    # Detect chains: a_chain /\ b_chain \/
    a_chain = ai_f[1:] == ai_p[:-1]
    b_chain = bi_p == bi_f
    a_f_smaller_p = diff_f[1:] < diff_p[:-1]
    b_p_smaller_f = diff_p < diff_f
    while len(np.nonzero(a_chain)[0]) or len(np.nonzero(b_chain)[0]):
        # print ".",
        # Chain /\/
        # If such a chain is found and the middle time is less
        # than the outer times, remove /-/
        # print_moj("  ","/\/ ", a_chain[:30]*b_chain[1:31])
        # print_moj("  ","/\/ ", a_chain[:30]*a_f_smaller_p[:30]*b_chain[1:31])
        # print_moj("  ","/\/ ", a_chain[:30]*a_f_smaller_p[:30]*b_chain[1:31]*(1-b_p_smaller_f[1:31]))
        i = np.nonzero(a_chain * a_f_smaller_p * b_chain[1:] * (1 - b_p_smaller_f[1:]))[0]
        ai_p[i] = -1
        bi_p[i] = -1
        ai_p[i + 1] = -1
        bi_p[i + 1] = -1
        # Chain \/\
        # If such a chain is found and the middle time is less
        # than the outer times, remove \-\
        # print_moj("","\/\ ", a_chain[:30]*b_chain[:30])
        # print_moj("","\/\ ", a_chain[:30]*(1-a_f_smaller_p[:30])*b_chain[:30])
        # print_moj("","\/\ ", a_chain[:30]*(1-a_f_smaller_p[:30])*b_chain[:30]*b_p_smaller_f[:30])
        i = np.nonzero(a_chain * (1 - a_f_smaller_p) * b_chain[:-1] * b_p_smaller_f[:-1])[0]
        ai_f[i] = -2
        bi_f[i] = -2
        ai_f[i + 1] = -2
        bi_f[i + 1] = -2
        # Chain /\-
        # If such a chain is found and the ending time is less
        # than the previous time, remove /--
        # print_moj("  ","/\- ", a_chain[:30]*(1-b_chain[1:31]))
        # print_moj("  ","/\- ", a_chain[:30]*a_f_smaller_p[:30]*(1-b_chain[1:31]))
        i = np.nonzero(a_chain * a_f_smaller_p * (1 - b_chain[1:]))[0]
        ai_p[i] = -1
        bi_p[i] = -1
        # Chain \/-
        # If such a chain is found and the ending time is less
        # than the previous time, remove \--
        # print_moj("","\/- ", (1-a_chain[:30])*b_chain[:30])
        # print_moj("","\/- ", (1-a_chain[:30])*b_chain[:30]*b_p_smaller_f[:30])
        i = np.nonzero((1 - a_chain) * b_chain[:-1] * b_p_smaller_f[:-1])[0]
        ai_f[i] = -2
        bi_f[i] = -2
        # Chain -\/
        # If such a chain is found and the starting time is less
        # than the following time, remove --/
        # print_moj("  ","-\/ ", (1-a_chain[:30])*b_chain[1:31])
        # print_moj("  ","-\/ ", (1-a_chain[:30])*b_chain[1:31]*(1-b_p_smaller_f[1:31]))
        i = np.nonzero((1 - a_chain) * b_chain[1:] * (1 - b_p_smaller_f[1:]))[0]
        ai_p[i + 1] = -1
        bi_p[i + 1] = -1
        # Chain -/\
        # If such a chain is found and the middle time is less
        # than the following time, remove --\
        # print_moj("","-/\ ", a_chain[:30]*(1-b_chain[:30]))
        # print_moj("","-/\ ", a_chain[:30]*(1-a_f_smaller_p[:30])*(1-b_chain[:30]))
        i = np.nonzero(a_chain * (1 - a_f_smaller_p) * (1 - b_chain[:-1]))[0]
        ai_f[i + 1] = -2
        bi_f[i + 1] = -2
        a_chain = ai_p[:-1] == ai_f[1:]
        b_chain = bi_p == bi_f
        # print "a_chain", a_chain
        # print "b_chain", b_chain
    return ai_f[ai_f > 0], bi_f[bi_f > 0]