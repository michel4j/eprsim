import numpy
from eprsim import SourceType, StationType, utils


class Source(SourceType):
    JITTER = 0.0

    def emit(self):
        s = 0.5
        v = utils.rand_unit_vec()
        p = 0.5 * numpy.sin(numpy.random.uniform(0, numpy.pi / 2)) ** 2
        n = s * 2
        return (*v, p, n), (*-v, p, n)


class Station(StationType):
    JITTER = 0.0

    def detect(self, setting, particle):
        h = particle[:3]
        p = particle[3]
        n = particle[4]
        s = n/2
        a = utils.rand_plane_vec(theta=setting/s)
        c = ((-1) ** n) * numpy.dot(h, a)
        if p <= abs(c):
            return self.time(), setting, numpy.sign(c)
        else:
            return self.time(), setting, numpy.nan
