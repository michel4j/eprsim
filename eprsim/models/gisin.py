# Symmetric Gisin & Gisin Model
# Phys.Lett. A260 (1999) 323-327 https://doi.org/10.48550/arXiv.quant-ph/9905018

import numpy
from eprsim import SourceType, StationType, utils


# random number generator
rng = numpy.random.default_rng()


class Source(SourceType):
    def emit(self):
        s = 0.5
        v = utils.rand_unit_vec()
        p = float(rng.choice([-1, 1]))
        return (*v, p, s), (*-v, -p, s)


class Station(StationType):
    def detect(self, setting, particle):
        h = particle[:3]
        p = particle[3]
        s = particle[4]
        a = utils.rand_plane_vec(theta=setting/s)
        c = ((-1) ** (2*s)) * numpy.dot(h, a)
        if p > 0 or numpy.random.uniform() <= abs(c):
            return self.time(), setting, numpy.sign(c)

