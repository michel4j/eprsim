# Naive model in which the particles carray the outcomes with them.
# Obviously does nothing interesting but can be used to verify that matching algorithms
# are working properly.

import numpy
from eprsim import SourceType, StationType


class Source(SourceType):
    def emit(self):
        e = float(numpy.random.choice([-1, 1]))
        return e, -e


class Station(StationType):
    def detect(self, setting, particle):
        c = numpy.sign(particle)
        return self.time(), setting, c
