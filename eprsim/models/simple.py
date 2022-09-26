import numpy
from eprsim import SourceType, StationType


class Source(SourceType):
    def emit(self):
        """
        λ = {e, p, s},   s = {1/2, 1}, p =  ½ sin²t, t ∈ [0..π/2), e ∈ [0..2π),  e' = e + 2πs
        """
        s = 0.5
        e = numpy.random.uniform(0, 2 * numpy.pi)
        p = 0.5 * numpy.sin(numpy.random.uniform(0, numpy.pi / 2)) ** 2
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
