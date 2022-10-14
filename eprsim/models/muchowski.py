# Contextual Model by Eugen Muchowski
# EPL, 134 (2021) 10004 www.epljournal.org
# doi: 10.1209/0295-5075/134/10004


import numpy
from eprsim import SourceType, StationType

# random number generator
rng = numpy.random.default_rng()


class Source(SourceType):
    def emit(self):
        phi = rng.uniform(0, 2 * numpy.pi)
        lamb = rng.uniform(0, 1)
        return (phi, lamb), (phi + numpy.pi/2, lamb)


class Station(StationType):
    def detect(self, setting, particle):
        phi, lamb  = particle
        delta = numpy.radians(setting - phi)
        return self.time(), setting, numpy.sign(numpy.cos(delta)**2 - lamb)

