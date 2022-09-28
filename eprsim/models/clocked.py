
import numpy

from eprsim import SourceType, StationType
from eprsim import utils


class Source(SourceType):
    def emit(self):
        v = utils.rand_unit_vec()
        s = 0.5
        p = numpy.random.uniform(0, 1)
        return (*v, p, s), (*-v, p, s)


class Station(StationType):
    """Detect a particle with a given/random setting"""

    def detect(self, setting, particle):
        h = particle[:3]
        p, s = particle[3:]
        n = 2*s

        svec = utils.rand_plane_vec(theta=setting)
        cn = ((-1) ** n) * numpy.dot(h, svec).sum()
        sn = numpy.linalg.norm(numpy.cross(h, svec))**2
        dt = sn / self.RATE
        return self.time() - dt, setting, numpy.sign(cn)
