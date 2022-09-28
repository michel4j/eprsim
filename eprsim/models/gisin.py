import numpy
from eprsim import SourceType, StationType, utils

# random number generator
rng = numpy.random.default_rng()


class Source(SourceType):
    JITTER = 0.0

    def emit(self):
        s = 0.5
        v = utils.rand_unit_vec()
        return (*v, s), (*-v, s)


class Station(StationType):
    JITTER = 0.0

    def detect(self, setting, particle):
        h = particle[:3]
        s = particle[3]
        a = utils.rand_plane_vec(theta=setting/s)
        c = ((-1) ** (2*s)) * numpy.dot(h, a)

        if rng.uniform() <= abs(c):
            return self.time(), setting, numpy.sign(c)
        else:
            return self.time(), setting, numpy.nan
