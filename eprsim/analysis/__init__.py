import functools
import itertools
from multiprocessing import Pool, cpu_count
from datetime import datetime

import numpy
import pandas
import scipy.stats
import tqdm

from prettytable import PrettyTable
from eprsim import utils

# random number generator
rng = numpy.random.default_rng()


MAX_MI_SIZE = 1000
MI_SAMPLES = 10000


def mutinf(xa, ya):
    i_xy = 0.0
    for x in numpy.unique(xa):
        for y in numpy.unique(ya):
            sel_x = (xa == x)
            sel_y = (ya == y)
            p_xy = (sel_x & sel_y).mean()
            if p_xy != 0.0:
                i_xy += p_xy * numpy.log10(p_xy/(sel_x.mean()*sel_y.mean()))
    return i_xy


def rand_mi1(x, y, i):
    """
    Calculate mutual information for a random dataset with equivalent probability distribution
    """
    rng.shuffle(x)
    rng.shuffle(y)

    return mutinf(x, y)


def rand_mi(x, y, i):
    xvals, xcounts = numpy.unique(x, return_counts=True)
    yvals, ycounts = numpy.unique(y, return_counts=True)
    return mutinf(
        rng.choice(xvals, p=xcounts/xcounts.sum(), size=MAX_MI_SIZE),
        rng.choice(yvals, p=ycounts/ycounts.sum(), size=MAX_MI_SIZE)
    )

def fmt_ang(angle):
    """
    Humanize test angles
    :param angle: angle in degrees
    :return: String representation of angle in radians
    """
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


def mi_signif(x, y, label="i,j"):
    mi = mutinf(x, y)
    pool = Pool(cpu_count())
    rand_mis = list(
        tqdm.tqdm(pool.imap(functools.partial(rand_mi, x, y), range(MI_SAMPLES)), total=MI_SAMPLES)
    )
    rank = scipy.stats.percentileofscore(rand_mis, mi)
    pct_95 = numpy.percentile(rand_mis, 95)
    avg_mi = numpy.mean(rand_mis)
    sigma_mi = numpy.std(rand_mis)
    return [
        f"I({label})",
        f"{mi:0.2e}",
        f"{rank:0.0f}",
        f"{pct_95:0.2e}",
        f"{avg_mi:0.2e}",
        f"{sigma_mi:0.2e}"
    ]


def analyse(alice, bob, spin=0.5):
    """EPRB Perform analysis"""

    # Find all settings used in simulation
    a1, a2 = sorted(numpy.unique(alice["setting"]))[:2]
    b1, b2 = sorted(numpy.unique(bob["setting"]))[:2]

    run_analysis(alice, bob, alice, bob, settings=(a1, b1, a2, b2), spin=spin)


def analyse_cfd(alice, bob, cindy, dave, spin):
    """EPRB Perform analysis with counterfactuals"""

    # Find all settings used in simulation
    a1 = sorted(numpy.unique(alice[:,1]))[0]
    a2 = sorted(numpy.unique(cindy[:,1]))[0]
    b1 = sorted(numpy.unique(bob[:,1]))[0]
    b2 = sorted(numpy.unique(dave[:,1]))[0]

    run_analysis(alice, bob, cindy, dave, settings=(a1, b1, a2, b2), spin=spin)


def run_analysis(*results, settings=None, spin=0.5, digits=3):

    """
    Core of analysis
    :param results: Data for experiment stations (A, B, C, D)
    :param settings: settings for CHSH test (a, b, c, d)
    :param c: Data for experiment station C
    :param d: Data for experiment station D
    :param spin: spin of particles, default 1/2
    :param digits: deciman places for correlation output
    """

    data = [
        (results[0], 'A', settings[0], 'a₁'),
        (results[1], 'B', settings[1], 'b₁'),
        (results[2 % len(results)], 'C', settings[2], 'a₂'),    # re-use alice and bob results if only two datasets
        (results[3 % len(results)], 'D', settings[3], 'b₂'),    # are provided. for CFD provide 4.
    ]

    eab_sim = numpy.zeros(4)   # E(a,b)
    nab_sim = numpy.zeros(4, dtype=int)   # N(a,b)
    eab_qm = numpy.zeros(4)    # E(a,b)_qm
    eab_err = numpy.zeros(4)   # Standard Error

    corr_tbl = PrettyTable()
    mi_tbl = PrettyTable()
    corr_tbl.align = "r"
    mi_tbl.align = "r"

    print("="*80)
    print("EPRB-Simulation Analysis")
    print("-"*80)
    print(f"a₁ = {fmt_ang(data[0][2])}, a₂ = {fmt_ang(data[2][2])}, b₁ = {fmt_ang(data[1][2])}, b₂ = {fmt_ang(data[3][2])}")

    corr_tbl.field_names = ["E(a,b)", "Nab", "<AB>sim", "<AB>qm", "Err"]
    mi_tbl.field_names = ["", "I(x,y)", "Rank", "95% Percentile", "<I>", "σI"]

    for i, pair in enumerate(itertools.product((data[0], data[2]), (data[1], data[3]))):
        (a_data, a_name, a_var, a_var_name), (b_data, b_name, b_var, b_var_name) = pair
        eab_qm[i] = qm_func(numpy.radians(b_var - a_var), spin)
        sel = (a_data["setting"] == a_var) & (b_data["setting"] == b_var)
        nab_sim[i] = sel.sum()

        if nab_sim[i] > 0:
            eab_sim[i] = (a_data["outcome"][sel] * b_data["outcome"][sel]).mean()
            eab_err[i] = numpy.abs(eab_sim[i] / numpy.sqrt(nab_sim[i]))

        corr_tbl.add_row([
            f'E({a_var_name},{b_var_name})',
            round(nab_sim[i], digits),
            round(eab_sim[i], digits),
            round(eab_qm[i], digits),
            round(eab_err[i], digits),
        ])
        # if i == 0:
        #     mi_tbl.add_rows([
        #         mi_signif(a_data['setting'], b_data['setting'], label=f'{a_var_name},{b_var_name}'),
        #         mi_signif(a_data['setting'], a_data['outcome'], label=f'{a_var_name},{a_name}'),
        #         mi_signif(b_data['setting'], b_data['outcome'], label=f'{b_var_name},{b_name}'),
        #         mi_signif(a_data['setting'], b_data['outcome'], label=f'{a_var_name},{b_name}'),
        #         mi_signif(b_data['setting'], a_data['outcome'], label=f'{b_var_name},{a_name}'),
        #         mi_signif(a_data['outcome'], b_data['outcome'], label=f'{a_name},{b_name}'),
        #     ])

    print()
    print(corr_tbl)
    chsh_sim = abs(eab_sim[0] - eab_sim[1] + eab_sim[2] + eab_sim[3])
    chsh_qm = abs(eab_qm[0] - eab_qm[1] + eab_qm[2] + eab_qm[3])
    print(f"CHSH: <= 2.0, Sim: {chsh_sim:0.3f}, QM: {chsh_qm.sum():0.3f}")
    print()
    # print(mi_tbl)
    # print()


def qm_func(a, spin=1/2):
    """
    Quantum mechanical Correlation for specific angular difference
    :param a: angle difference in radians
    :param spin: spin of particles, default 1/2
    """
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


