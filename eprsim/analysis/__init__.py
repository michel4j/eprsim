import h5py
import hdf5plugin
import numpy


class Analysis(object):
    def __init__(self, alice_file, bob_file):
        with h5py.File(alice_file, 'r') as f:
            self.alice_raw = f['data'][:]
        with h5py.File(bob_file, 'r') as f:
            self.bob_raw = f['data'][:]
        ai, bi = self.match(self.alice_raw[:,0], self.bob_raw[:,0])
        self.alice = self.alice_raw[ai]
        self.bob = self.bob_raw[bi]

    def match(self, a_times, b_times):
        # find Alice's time-tags that occur just before Bob's time-tags
        a2b_i = numpy.searchsorted(b_times, a_times)

        # Remove duplicates and get candidate matching indices on both sides
        # bi_fw contains Bob's index of the detection
        # following the ones that have the index ai_fw at Alice.
        ai_fw = numpy.nonzero(numpy.diff(a2b_i))[0]
        ai_bw = ai_fw + 1
        bi_fw = a2b_i[ai_fw]
        bi_bw = a2b_i[ai_bw] - 1

        # Time difference calculation
        dt_fw = b_times[bi_fw] - a_times[ai_fw]
        dt_bw = a_times[ai_bw] - b_times[bi_bw]

        # Link notation below: \ a detection at Bob follows a detection at
        # Alice, / a detection at Bob precedes a detection at Alice.
        # Detect chains: a_chain /\ b_chain \/

        a_chain = ai_fw[1:] == ai_bw[:-1]
        b_chain = bi_bw == bi_fw
        a_f_smaller_b = dt_fw[1:] < dt_bw[:-1]
        b_p_smaller_f = dt_bw < dt_fw

        while len(numpy.nonzero(a_chain)[0]) or len(numpy.nonzero(b_chain)[0]):
            # print ".",
            # Chain /\/
            # If such a chain is found and the middle time is less
            # than the outer times, remove /-/
            # print_moj("  ","/\/ ", a_chain[:30]*b_chain[1:31])
            # print_moj("  ","/\/ ", a_chain[:30]*a_f_smaller_b[:30]*b_chain[1:31])
            # print_moj("  ","/\/ ", a_chain[:30]*a_f_smaller_b[:30]*b_chain[1:31]*(1-b_p_smaller_f[1:31]))
            i = numpy.nonzero(a_chain * a_f_smaller_b * b_chain[1:] * (1 - b_p_smaller_f[1:]))[0]
            ai_bw[i] = -1
            bi_bw[i] = -1
            ai_bw[i + 1] = -1
            bi_bw[i + 1] = -1
            # Chain \/\
            # If such a chain is found and the middle time is less
            # than the outer times, remove \-\
            # print_moj("","\/\ ", a_chain[:30]*b_chain[:30])
            # print_moj("","\/\ ", a_chain[:30]*(1-a_f_smaller_b[:30])*b_chain[:30])
            # print_moj("","\/\ ", a_chain[:30]*(1-a_f_smaller_b[:30])*b_chain[:30]*b_p_smaller_f[:30])
            i = numpy.nonzero(a_chain * (1 - a_f_smaller_b) * b_chain[:-1] * b_p_smaller_f[:-1])[0]
            ai_fw[i] = -2
            bi_fw[i] = -2
            ai_fw[i + 1] = -2
            bi_fw[i + 1] = -2
            # Chain /\-
            # If such a chain is found and the ending time is less
            # than the previous time, remove /--
            # print_moj("  ","/\- ", a_chain[:30]*(1-b_chain[1:31]))
            # print_moj("  ","/\- ", a_chain[:30]*a_f_smaller_b[:30]*(1-b_chain[1:31]))
            i = numpy.nonzero(a_chain * a_f_smaller_b * (1 - b_chain[1:]))[0]
            ai_bw[i] = -1
            bi_bw[i] = -1
            # Chain \/-
            # If such a chain is found and the ending time is less
            # than the previous time, remove \--
            # print_moj("","\/- ", (1-a_chain[:30])*b_chain[:30])
            # print_moj("","\/- ", (1-a_chain[:30])*b_chain[:30]*b_p_smaller_f[:30])
            i = numpy.nonzero((1 - a_chain) * b_chain[:-1] * b_p_smaller_f[:-1])[0]
            ai_fw[i] = -2
            bi_fw[i] = -2
            # Chain -\/
            # If such a chain is found and the starting time is less
            # than the following time, remove --/
            # print_moj("  ","-\/ ", (1-a_chain[:30])*b_chain[1:31])
            # print_moj("  ","-\/ ", (1-a_chain[:30])*b_chain[1:31]*(1-b_p_smaller_f[1:31]))
            i = numpy.nonzero((1 - a_chain) * b_chain[1:] * (1 - b_p_smaller_f[1:]))[0]
            ai_bw[i + 1] = -1
            bi_bw[i + 1] = -1
            # Chain -/\
            # If such a chain is found and the middle time is less
            # than the following time, remove --\
            # print_moj("","-/\ ", a_chain[:30]*(1-b_chain[:30]))
            # print_moj("","-/\ ", a_chain[:30]*(1-a_f_smaller_b[:30])*(1-b_chain[:30]))
            i = numpy.nonzero(a_chain * (1 - a_f_smaller_b) * (1 - b_chain[:-1]))[0]
            ai_fw[i + 1] = -2
            bi_fw[i + 1] = -2
            a_chain = ai_bw[:-1] == ai_fw[1:]
            b_chain = bi_bw == bi_fw
            # print "a_chain", a_chain
            # print "b_chain", b_chain

        return ai_fw[ai_fw > 0], bi_fw[bi_fw > 0]

    def match1(self, a_times, b_times):
        a_idx = numpy.array([], dtype=int)
        b_idx = numpy.array([], dtype=int)
        i = numpy.searchsorted(a_times, b_times[0])
        j = numpy.searchsorted(b_times, a_times[0])
        step = min(min(1000, (a_times.shape[0] - i)), min(1000, (b_times.shape[0] - j)))
        while step > 0:
            ai, bi = find_matches(a_times[i:i+step], b_times[j:j+step])
            p = a_idx.size
            a_idx.resize(a_idx.size + ai.size)
            b_idx.resize(b_idx.size + bi.size)
            a_idx[p:] = ai
            b_idx[p:] = bi
            j += step - 10  # overlap by 10 items
            i += step - 10
            step = min(min(1000, (a_times.shape[0] - i)), min(1000, (b_times.shape[0] - j)))
        return a_idx, b_idx

    def mutual(self, X, Y):
        I_XY = 0.0
        for x in set(X):
            for y in set(Y):
                sel_x = (X == x)
                sel_y = (Y == y)
                p_xy = (sel_x & sel_y).mean()
                if p_xy > 0:
                    I_XY += p_xy * numpy.log10(p_xy / (sel_x.mean() * sel_y.mean()))
        return I_XY


def find_matches(a, b):
    dist = numpy.abs(b - a.reshape(-1, 1))
    a1, b1 = numpy.unique(numpy.argmin(dist, axis=1), return_index=True)
    b2, a2 = numpy.unique(numpy.argmin(dist, axis=0), return_index=True)
    if a1.shape > a2.shape:
        return a1, b1
    else:
        return a2, b2
