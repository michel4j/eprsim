import numpy
from eprsim import SourceType, StationType


class Source(SourceType):
    def emit(self):
        s = 0.5
        e = numpy.random.uniform(0, 2 * numpy.pi)
        p = (2/numpy.sqrt(1 + 3*numpy.random.uniform())) - 1
        return (e, p, s), (e + s * numpy.pi, p, s)


class Station(StationType):
    def detect(self, setting, particle):
        e, p, s = particle
        a = numpy.radians(setting)
        n = 2*s
        c = (-1 ** n) * numpy.cos((a - e)/s)
        if p <= abs(c):
            return self.time(), setting, numpy.sign(c)
        else:
            return self.time(), setting, numpy.nan
