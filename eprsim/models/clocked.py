# Variant of EPR-Clocked simulation using vectors, ported to new framework
# Michel Fodje -- 2013

import numpy

from eprsim import SourceType, StationType
from eprsim import utils


# random number generator
rng = numpy.random.default_rng()


class Source(SourceType):
    def emit(self):
        v = utils.rand_unit_vec()
        p = 0.5 * numpy.sin(numpy.random.uniform(0, numpy.pi / 2)) ** 2
        s = 1/2
        return (*v, p, s), (*-v, p, s)


class Station(StationType):
    TIME_SCALE = 2e-4

    def detect(self, setting, particle):
        h = particle[:3]
        p, s = particle[3:]

        a = utils.rand_plane_vec(theta=setting/s)
        c = ((-1) ** (2*s)) * numpy.dot(h, a)
        dt = self.TIME_SCALE * max((p - abs(c)), 0.0)

        return self.time() + dt, setting, numpy.sign(c)

