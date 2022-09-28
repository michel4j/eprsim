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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        y = numpy.abs(numpy.cos(numpy.linspace(0, numpy.pi, 1_000_000)))
        self.generator = utils.pdf_sampler(y)

    def detect(self, setting, particle):
        h = particle[:3]
        s = particle[3]
        a = utils.rand_plane_vec(theta=setting/s)
        c = ((-1) ** (2*s)) * numpy.dot(h, a)

        if self.generator.rvs() <= abs(c):
            return self.time(), setting, numpy.sign(c)
        else:
            return self.time(), setting, numpy.nan
