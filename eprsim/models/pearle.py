# Revisited Pearle Mode by R.D Gill, inspired by EPR-Simple
# Entropy 2020, 22(1), 1
# https://doi.org/10.3390%2Fe22010001
# https://doi.org/10.1103/PhysRevD.2.1418

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
        c = (-1 ** (2*s)) * numpy.cos((a - e)/s)
        if p <= abs(c):
            return self.time(), setting, numpy.sign(c)
        else:
            return self.time(), setting, numpy.nan
