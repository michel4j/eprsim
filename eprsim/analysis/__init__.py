import functools
import itertools
from collections import Counter
from multiprocessing import Pool, cpu_count
from datetime import datetime

import numpy
import pandas
import scipy.interpolate as interpolate
import scipy.stats
import tqdm

from prettytable import PrettyTable
from eprsim import utils


def inverse_transform_sampling(data, n_bins=40, n_samples=1000):
    hist, bin_edges = numpy.histogram(data, bins=n_bins, density=True)
    cum_values = numpy.zeros(bin_edges.shape)
    cum_values[1:] = numpy.cumsum(hist * numpy.diff(bin_edges))
    inv_cdf = interpolate.interp1d(cum_values, bin_edges)
    r = numpy.random.rand(n_samples)
    return inv_cdf(r)


def sample_pdf(func, low=0, high=1, size=None):
    edges = numpy.linspace(low, high, 1000000)
    x = edges[:-1] + numpy.diff(edges) / 2
    y = (func(x) * 1e6).astype(int)

    dist = scipy.stats.rv_histogram((y * 1e6))
    return dist.rvs(size=size)


def randmut(xvals, xprobs, yvals, yprobs, size, i):
    return mutinf(
        numpy.random.choice(xvals, p=xprobs, size=size),
        numpy.random.choice(yvals, p=yprobs, size=size)
    )


def tmat(X, order=1):
    vals = sorted(set(X))
    mat = numpy.zeros((len(vals), len(vals)))
    for (x, y), c in Counter(zip(X, X[order:])).iteritems():
        mat[vals.index(x), vals.index(y)] = c
    return vals, mat / mat.sum()


def mi_sig(X, Y, names=("i", "j")):
    mi = mutinf(X, Y)
    pool = Pool(cpu_count())
    u_x, c_x = numpy.unique(X, return_counts=True)
    u_y, c_y = numpy.unique(Y, return_counts=True)
    print(u_x, c_x, u_y, c_y)
    size = c_x.sum()
    xprobs, yprobs = c_x / size, c_y / size
    rmis = list(
        tqdm.tqdm(pool.imap(functools.partial(randmut, u_x, xprobs, u_y, yprobs, size), range(1000)), total=1000))
    print(
        f"I({names[0]},{names[1]})={mi:8.6f}   pct={scipy.stats.percentileofscore(rmis, mi):8.3f}   "
        f"95%={numpy.percentile(rmis, 95):8.6f}   <I>={numpy.mean(rmis):8.6f}   \u03C3={numpy.std(rmis):8.6f}"
    )


def mi_kernel(X, Y, xy):
    x, y = xy
    sel_x = (X == x)
    sel_y = (Y == y)
    p_xy = (sel_x & sel_y).mean()
    return p_xy * numpy.log10(p_xy / (sel_x.mean() * sel_y.mean()))


class MIKernel(object):
    def __init__(self, X, Y):
        xyvals, xycounts = numpy.unique(numpy.column_stack((X, Y)), return_counts=True, axis=0)
        xvals, xcounts = numpy.unique(X, return_counts=True)
        yvals, ycounts = numpy.unique(Y, return_counts=True)
        self.xprobs = {key: prob for key, prob in zip(xvals, xcounts / xcounts.sum())}
        self.yprobs = {key: prob for key, prob in zip(yvals, ycounts / ycounts.sum())}
        self.xyprobs = {tuple(key): prob for key, prob in zip(xyvals, xycounts / xycounts.sum())}
        self.pairs = list(itertools.product(xvals, yvals))

    def __call__(self, xy):
        x, y = xy
        pxy = self.xyprobs.get((x, y), 0.0)
        px = self.xprobs.get(x, 0.0)
        py = self.yprobs.get(y, 0.0)
        denom = px * py

        if pxy > 0 and denom > 0.0:
            return pxy * numpy.log(pxy / denom)
        else:
            return 0.0


def mutinf(X, Y):
    kernel = MIKernel(X, Y)
    return sum(
        map(kernel, kernel.pairs)
    )


def fmt(angle):
    return {
        22.5: "π/8",
        30.0:  "π/6",
        45.0:  "π/4",
        60.0:  "π/3",
        67.5: "3π/8",
        90.0:  "π/2",
        135.0: "3π/4",
        180.0: "π",
        270.0: "3π/2",
        360.0: "0 ",
    }.get(round(angle,1), f"{angle:g}")


def analyse(alice, bob, spin=0.5):
    """EPRB Perform analysis"""

    # Find all settings used in simulation
    a1, a2 = sorted(numpy.unique(alice["setting"]))[:2]
    b1, b2 = sorted(numpy.unique(bob["setting"]))[:2]

    data = [
        (alice, 'A', a1, 'a₁'),
        (bob, 'B', b1, 'b₁'),
        (alice, 'C', a2, 'a₂'),
        (bob, 'D', b2, 'b₂'),
    ]

    run_analysis(data, spin)

    print('Mutual Information')
    print('-'*80)
    print('I(a,A)  = {:0.5f} bits'.format(mutinf(alice['setting'], alice['outcome'])))
    print('I(b,B)  = {:0.5f} bits'.format(mutinf(bob['setting'], bob['outcome'])))
    print('I(a,b)  = {:0.5f} bits'.format(mutinf(alice['setting'], bob['setting'])))
    print('I(A,B)  = {:0.5f} bits'.format(mutinf(alice['outcome'], bob['outcome'])))
    print('='*80)


def analyse_cfd(alice, bob, cindy, dave, spin):
    """EPRB Perform analysis with counterfactuals"""

    # Find all settings used in simulation
    a1 = sorted(numpy.unique(alice[:,1]))[0]
    a2 = sorted(numpy.unique(cindy[:,1]))[0]
    b1 = sorted(numpy.unique(bob[:,1]))[0]
    b2 = sorted(numpy.unique(dave[:,1]))[0]

    data = [
        (alice, 'A', a1, 'a₁'),
        (bob, 'B', b1, 'b₁'),
        (cindy, 'C', a2, 'a₂'),
        (dave, 'D', b2, 'b₂'),
    ]

    run_analysis(data, spin)

    print('Mutual Information')
    print('-'*80)
    print('I(a,A)  = {:0.5f} bits'.format(mutinf(alice['setting'], alice['outcome'])))
    print('I(b,B)  = {:0.5f} bits'.format(mutinf(bob['setting'], bob['outcome'])))
    print('I(a,b)  = {:0.5f} bits'.format(mutinf(alice['setting'], bob['setting'])))
    print('I(A,B)  = {:0.5f} bits'.format(mutinf(alice['outcome'], bob['outcome'])))
    print('='*80)


def run_analysis(data, spin=0.5, digits=3):
    eab_sim = numpy.zeros(4)   # E(a,b)
    nab_sim = numpy.zeros(4, dtype=int)   # N(a,b)
    eab_qm = numpy.zeros(4)    # E(a,b)_qm
    eab_err = numpy.zeros(4)   # Standard Error

    table = PrettyTable()
    table.align = "r"

    print("="*80)
    print("EPRB-Simulation Analysis")
    print("-"*80)
    print(f"a₁ = {fmt(data[0][2])}, a₂ = {fmt(data[2][2])}, b₁ = {fmt(data[1][2])}, b₂ = {fmt(data[3][2])}")

    table.field_names = ["E(a,b)", "Nab", "<AB>sim", "<AB>qm", "Err"]

    for i, pair in enumerate(itertools.product((data[0], data[2]), (data[1], data[3]))):
        (a_data, a_name, a_var, a_var_name), (b_data, b_name, b_var, b_var_name) = pair
        e_name = f'E({a_var_name},{b_var_name})'
        eab_qm[i] = qm_func(numpy.radians(b_var - a_var), spin)
        sel = (a_data["setting"] == a_var) & (b_data["setting"] == b_var)
        nab_sim[i] = sel.sum()

        if nab_sim[i] > 0:
            eab_sim[i] = (a_data["outcome"][sel] * b_data["outcome"][sel]).mean()
            eab_err[i] = numpy.abs(eab_sim[i] / numpy.sqrt(nab_sim[i]))

        table.add_row([
            f'E({a_var_name},{b_var_name})',
            round(nab_sim[i], digits),
            round(eab_sim[i], digits),
            round(eab_qm[i], digits),
            round(eab_err[i], digits),
        ])

    print()
    print(table)
    print()

    chsh_sim = abs(eab_sim[0] - eab_sim[1] + eab_sim[2] + eab_sim[3])
    chsh_qm = abs(eab_qm[0] - eab_qm[1] + eab_qm[2] + eab_qm[3])

    print(f"CHSH: <= 2.0, Sim: {chsh_sim:0.3f}, QM: {chsh_qm.sum():0.3f}")
    print()


def qm_func(a, spin):
    n = 1/spin
    return (-1 ** n) * numpy.cos(n * a)


class Analysis(object):
    def __init__(self, name=None, dt=1e-5):
        if name is None:
            name = datetime.now().strftime('%y%m%dT%H')
        alice = f'A-{name}.h5'
        bob = f'B-{name}.h5'

        # remove missing outcomes and keep one outcome per pulse window for pulsed experiments.
        self.alice_df = pandas.read_hdf(alice).dropna(subset='outcome').drop_duplicates(subset='time', keep='first')
        self.bob_df = pandas.read_hdf(bob).dropna(subset='outcome').drop_duplicates(subset='time', keep='first')
        self.alice_raw = self.alice_df.to_records(index=False)
        self.bob_raw = self.bob_df.to_records(index=False)

        ai, bi = utils.match(self.alice_raw['time'], self.bob_raw['time'])
        self.alice = self.alice_raw[ai]
        self.bob = self.bob_raw[bi]
        sel = numpy.abs(self.bob['time'] - self.alice['time']) < dt
        analyse(self.alice[sel], self.bob[sel])


def find_matches(a, b):
    dist = numpy.abs(b - a.reshape(-1, 1))
    a1, b1 = numpy.unique(numpy.argmin(dist, axis=1), return_index=True)
    b2, a2 = numpy.unique(numpy.argmin(dist, axis=0), return_index=True)
    if a1.shape > a2.shape:
        return a1, b1
    else:
        return a2, b2
