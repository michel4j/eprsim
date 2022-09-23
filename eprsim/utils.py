
import numpy



def rand_unit_vec(size=None, theta=None):
    """
    Generate a random unit vector uniformly distributed on a sphere
    """
    theta = numpy.random.uniform(0, numpy.pi*2, size=size) if theta is None else numpy.radians(theta)
    u = numpy.random.uniform(-1, 1, size=size)
    v = numpy.sqrt(1-u**2)
    if size:
        return numpy.array([v*numpy.cos(theta), v*numpy.sin(theta), u]).T
    else:
        return numpy.array([v*numpy.cos(theta), v*numpy.sin(theta), u])


def rand_plane_vec(size=None, theta=None):
    """
    Generate a random unit vector uniformly distributed on a circle
    """
    theta = numpy.random.uniform(0, numpy.pi*2, size=size) if theta is None else numpy.radians(theta)
    u = 0
    v = numpy.sqrt(1-u**2)
    if size:
        return numpy.array([v*numpy.cos(theta), v*numpy.sin(theta), numpy.array([u]*size)]).T
    else:
        return numpy.array([v*numpy.cos(theta), v*numpy.sin(theta), u])


