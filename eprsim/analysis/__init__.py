import functools
import itertools
from multiprocessing import Pool, cpu_count

import numpy
import pandas
import scipy.stats
from tqdm import tqdm
import matplotlib
from prettytable import PrettyTable
from eprsim import utils

matplotlib.use('Gtk3Agg')
from matplotlib import pyplot as plt

# random number generator
rng = numpy.random.default_rng()

MAX_MI_SIZE = 1_000
MI_SAMPLES = 10_000


def mutinf(xa, ya):
    """
    Calculate mutual information
    """
    ixy = 0.0
    for x in numpy.unique(xa):
        sel_x = (xa == x)
        px = sel_x.mean()
        for y in numpy.unique(ya):
            sel_y = (ya == y)
            pxy = (sel_x & sel_y).mean()
            if pxy != 0.0:
                ixy += pxy * numpy.log10(pxy / (px * sel_y.mean()))
    return ixy


def rand_mi(x, y, i):
    """
    Calculate mutual information for a random dataset with equivalent probability distribution
    """
    return mutinf(rng.permutation(x), rng.permutation(y))


def rand_mi_size(X, Y, size, i):
    uX, cX = numpy.unique(X, return_counts=True)
    uY, cY = numpy.unique(Y, return_counts=True)
    total = cY.sum()
    xprobs, yprobs = cX / total, cY / total
    return mutinf(
        rng.choice(uX, p=xprobs, size=size),
        rng.choice(uY, p=yprobs, size=size)
    )


def rand_mi_pdf(x_vals, y_vals, x_probs, y_probs, size, i):
    return mutinf(
        rng.choice(x_vals, p=x_probs, size=size),
        rng.choice(y_vals, p=y_probs, size=size)
    )


def fmt_ang(angle):
    """
    Humanize test angles
    :param angle: angle in degrees
    :return: String representation of angle in radians
    """
    return {
        22.5: "π/8",
        30.0: "π/6",
        45.0: "π/4",
        60.0: "π/3",
        67.5: "3π/8",
        90.0: "π/2",
        135.0: "3π/4",
        180.0: "π",
        270.0: "3π/2",
        360.0: "0 ",
    }.get(round(angle, 1), f"{angle:g}")


def mi_signif(x, y, label="i,j"):
    mi = mutinf(x, y)
    with Pool(cpu_count()) as pool:
        rand_mis = numpy.empty((MI_SAMPLES,))

        x_vals, x_counts = numpy.unique(x, return_counts=True)
        y_vals, y_counts = numpy.unique(y, return_counts=True)
        total = x_counts.sum()
        x_probs, y_probs = x_counts / total, y_counts / total

        work = pool.imap_unordered(functools.partial(rand_mi_pdf, x_vals, y_vals, x_probs, y_probs, MAX_MI_SIZE),
                                   range(MI_SAMPLES))
        for i, out in tqdm(enumerate(work), total=MI_SAMPLES, leave=False, ncols=80):
            rand_mis[i] = out
        rank = scipy.stats.percentileofscore(rand_mis, mi)
        pct_95 = numpy.percentile(rand_mis, 95)
        avg_mi = numpy.mean(rand_mis)
        sigma_mi = numpy.std(rand_mis)
        print('\r', end="")

    return [
        f"I({label})",
        f"{mi:0.2e}",
        f"{rank:0.0f}",
        f"{pct_95:0.2e}",
        f"{avg_mi:0.2e}",
        f"{sigma_mi:0.2e}"
    ]


def analyse(alice, bob, spin=0.5):
    """
    EPRB Perform analysis
    """
    settings = 0.0, 22.5, 45, 67.5
    chsh_analysis(alice, bob, alice, bob, settings=settings, spin=spin)


def analyse_cfd(alice, bob, cindy, dave, spin):
    """
    EPRB Perform analysis with counterfactuals
    """

    settings = 0.0, 22.5, 45, 67.5
    chsh_analysis(alice, bob, cindy, dave, settings=settings, spin=spin)


def chsh_analysis(*results, settings=None, spin=0.5, digits=3):
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
        (results[2 % len(results)], 'C', settings[2], 'a₂'),  # re-use alice and bob results if only two datasets
        (results[3 % len(results)], 'D', settings[3], 'b₂'),  # are provided. for CFD provide 4.
    ]

    eab_sim = numpy.zeros(4)  # E(a,b)
    nab_sim = numpy.zeros(4, dtype=int)  # N(a,b)
    eab_qm = numpy.zeros(4)  # E(a,b)_qm
    eab_err = numpy.zeros(4)  # Standard Error

    corr_tbl = PrettyTable()

    corr_tbl.align = "r"
    corr_tbl.field_names = ["E(a,b)", "Nab", "<AB>sim", "<AB>qm", "Err"]

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

    print("=" * 80)
    print("EPRB-Simulation Analysis")
    print("-" * 80)
    print(
        f"a₁ = {fmt_ang(data[0][2])}, a₂ = {fmt_ang(data[2][2])}, b₁ = {fmt_ang(data[1][2])}, b₂ = {fmt_ang(data[3][2])}")
    print(corr_tbl)
    chsh_sim = abs(eab_sim[0] - eab_sim[1] + eab_sim[2] + eab_sim[3])
    chsh_qm = abs(eab_qm[0] - eab_qm[1] + eab_qm[2] + eab_qm[3])
    print(f"CHSH: <= 2.0, Sim: {chsh_sim:0.3f}, QM: {chsh_qm.sum():0.3f}")
    print()

    # Calculate correlation curve, express angle difference in range 0 deg to pi range only.
    d_ang = numpy.abs(results[0]['setting'] - results[1]['setting']) % 180.
    corr_exp = results[0]['outcome'] * results[1]['outcome']
    angles, n_ab = numpy.unique(d_ang, return_counts=True)
    exp_ab = numpy.zeros_like(angles)
    for i, a in enumerate(angles):
        sel = (d_ang == a)
        exp_ab[i] = corr_exp[sel].mean()
    exp_qm = qm_func(numpy.radians(angles), spin=spin)

    plt.plot(angles, exp_ab, 'm-x', label='Model: E(a,b)', lw=0.5)
    plt.plot(angles, exp_qm, 'b-+', label='QM', lw=0.5)
    plt.legend()
    plt.xlim(0, 180)
    plt.savefig('correlations.png')
    plt.clf()


def ch_analysis(orig, coinc, settings=None):
    """
    Core of analysis
    :param results: Data for experiment stations (A, B, C, D)
    :param settings: settings for CHSH test (a, b, c, d)

    """

    a1, b1, a2, b2 = settings

    c11 = (
        (coinc['Alice']['setting'] == a1) &
        (coinc['Bob']['setting'] == b1) &
        (coinc['Alice']['outcome'] == 1) &
        (coinc['Bob']['outcome'] == 1)
    ).sum()

    c12 = (
        (coinc['Alice']['setting'] == a1) &
        (coinc['Bob']['setting'] == b2) &
        (coinc['Alice']['outcome'] == 1) &
        (coinc['Bob']['outcome'] == 1)
    ).sum()

    c22 = (
        (coinc['Alice']['setting'] == a2) &
        (coinc['Bob']['setting'] == b2) &
        (coinc['Alice']['outcome'] == 1) &
        (coinc['Bob']['outcome'] == 1)
    ).sum()

    c21 = (
        (coinc['Alice']['setting'] == a2) &
        (coinc['Bob']['setting'] == b1) &
        (coinc['Alice']['outcome'] == 1) &
        (coinc['Bob']['outcome'] == 1)
    ).sum()

    sa1 = (
        (coinc['Alice']['setting'] == a1) &
        (coinc['Bob']['setting'] == b1) &
        (coinc['Alice']['outcome'] == 1)
    ).sum()

    sb1 = (
        (coinc['Alice']['setting'] == a1) &
        (coinc['Bob']['setting'] == b1) &
        (coinc['Bob']['outcome'] == 1)
    ).sum()

    corr_tbl = PrettyTable()
    corr_tbl.align = "r"
    corr_tbl.field_names = ["Parameter", "Counts",]
    corr_tbl.add_row([f'C(a₁,b₁)', round(c11)])
    corr_tbl.add_row([f'C(a₁,b₂)', round(c12)])
    corr_tbl.add_row([f'C(a₂,b₁)', round(c21)])
    corr_tbl.add_row([f'C(a₂,b₂)', round(c22)])
    corr_tbl.add_row([f'S(a₁)', round(sa1)])
    corr_tbl.add_row([f'S(b₁)', round(sb1)])
    print(corr_tbl)
    return (c11 + c12 + c21 - c22)/(sa1 + sb1)


def qm_func(a, spin=0.5):
    """
    Quantum mechanical Correlation for specific angular difference
    :param a: angle difference in radians
    :param spin: spin of particles, default 1/2
    """
    n = 1 / spin
    return (-1 ** n) * numpy.cos(n * a)


class Analysis(object):
    def __init__(self):
        a_file, b_file = utils.get_latest('A-*.h5', 'B-*.h5')
        assert all((a_file, b_file)), "Matching Data files were not found in folder."
        print('Running analysis for data: ')
        print(f'\tAlice - {a_file}')
        print(f'\tBob   - {b_file}')

        # Original data, no manipulations
        self.original = {
            'Alice': pandas.read_hdf(a_file),
            'Bob': pandas.read_hdf(b_file)
        }

        # remove nan outcome values, ie particles seen by station but not detected
        self.detected = {
            name: data.dropna(subset='outcome')
            for name, data in self.original.items()
        }

        # keep only first event in a time batch, corresponds to pulsed measurements
        # with pre-agreed time slots. does nothing if time precision is smaller than
        # emission/detection period. Also convert to numpy record array at this point
        self.unique = {
            name: data.drop_duplicates(subset='time', keep='first').to_records(index=False)
            for name, data in self.detected.items()
        }

        # match data pairs and create final matched data for further analysis
        # coincidence time constraints have not been applied yet
        ai, bi = utils.match(self.unique['Alice']['time'], self.unique['Bob']['time'])
        self.data = {
            'Alice': self.unique['Alice'][ai],
            'Bob': self.unique['Bob'][bi]
        }

    def analyse(self, window=1e-15):
        sel = numpy.abs(self.data['Alice']['time'] - self.data['Bob']['time']) < window
        coinc = {
            'Alice': self.data['Alice'][sel],
            'Bob': self.data['Bob'][sel]
        }
        analyse(coinc['Alice'], coinc['Bob'])
        ch = ch_analysis(self.original, coinc, settings=(0.0, 22.5, 45, 67.5))
        print(f'CH = {ch: 0.4e}  <= 1')

    def mi_test(self, window=1e-15):
        sel = numpy.abs(self.data['Alice']['time'] - self.data['Bob']['time']) < window
        alice = self.data['Alice'][sel]
        bob = self.data['Bob'][sel]

        table = PrettyTable()
        table.align = "r"
        table.field_names = ["", "I(x,y)", "Rank", "95% Percentile", "<I>", "σI"]
        print()
        print("Mutual Information Analysis.")
        table.add_rows([
            mi_signif(alice['setting'], bob['setting'], label=f'a,b'),
            mi_signif(alice['setting'], alice['outcome'], label=f'a,A'),
            mi_signif(bob['setting'], bob['outcome'], label=f'b,B'),
            mi_signif(alice['outcome'], bob['outcome'], label=f'A,B'),
        ])
        print(table)

    def stats(self, window=1):
        table = PrettyTable()
        table.align = 'r'
        table.field_names = ["Station", "Seen", "Detected", "% Detected", "% Matched", "% Coinc"]
        print()
        print("Station event counts.")
        sel = numpy.abs(self.data['Alice']['time'] - self.data['Bob']['time']) < window
        coinc = sel.sum()

        for station in ['Alice', 'Bob']:
            seen = len(self.original[station])
            detected = len(self.detected[station])
            matched = len(self.data[station])
            table.add_row([
                f'{station} - All',
                seen,
                detected,
                f'{detected / seen:0.1%}',
                f'{matched / detected:0.1%}',
                f'{coinc / matched:0.1%}'
            ])

        for station in ['Alice', 'Bob']:
            for setting in numpy.unique(self.original[station]['setting']):
                seen = (self.original[station]['setting'] == setting).sum()
                detected = (self.detected[station]['setting'] == setting).sum()
                matched = (self.data[station]['setting'] == setting).sum()
                coinc = (self.data[station][sel]['setting'] == setting).sum()

                table.add_row([
                    f'{station} - {setting}',
                    seen,
                    detected,
                    f'{detected / seen:0.1%}',
                    f'{matched / detected:0.1%}',
                    f'{coinc / matched:0.1%}'
                ])
        print(table)
