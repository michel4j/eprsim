#  3D version of original EPR-Simple using vectors instead of angles
#  Michel Fodje -- 2013


import numpy
from eprsim import SourceType, StationType, utils


class Source(SourceType):
    def emit(self):
        s = 0.5
        v = utils.rand_unit_vec()
        p = 0.5 * numpy.sin(numpy.random.uniform(0, numpy.pi / 2)) ** 2
        return (*v, p, s), (*-v, p, s)


class Station(StationType):
    def detect(self, setting, particle):
        h = particle[:3]
        p = particle[3]
        s = particle[4]
        a = utils.rand_plane_vec(theta=setting/s)
        c = ((-1) ** (2*s)) * numpy.dot(h, a)
        if p <= abs(c):
            return self.time(), setting, numpy.sign(c)
        else:
            return self.time(), setting, numpy.nan
