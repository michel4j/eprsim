import numpy
from datetime import datetime
from eprsim import SourceType, StationType

numpy.seterr(divide='ignore')


class Source(SourceType):
    def emit(self):
        """
        λ = {e, p, s},   s = {1/2, 1}, p =  ½ sin²t, t ∈ [0..π/2), e ∈ [0..2π),  e' = e + 2πs
        """
        s = 0.5
        e = numpy.random.uniform(0, 2*numpy.pi)
        p = 0.5 * numpy.sin(numpy.random.uniform(0, numpy.pi/2))**2
        return (e, p, s), (e+numpy.pi, p, s)


class Station(StationType):
    def detect(self, setting, particle):
        e, p, s = particle
        n = 2*s
        c = numpy.sign(numpy.abs(numpy.cos(n*(setting-e))) - p)
        o = numpy.sign((-1**n) * numpy.cos(n*(setting-e)))
        return max(0, c), (self.time(), setting, o)
