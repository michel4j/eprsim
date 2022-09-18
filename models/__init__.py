#
#    Simple EPR simulation
#    Copyright (C) 2022  Michel Fodje
#
#    This program is free software; you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation; either version 2 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
import random
#    You should have received a copy of the GNU General Public License along
#    with this program; if not, write to the Free Software Foundation, Inc.,
#    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.


import time
import zmq
import msgpack
import h5py


class Source(object):
    """Generate and emit two particles with hidden variables"""

    def __init__(self, port=10001):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind("tcp://*:%s" % port)
        self.running = False

    def emit(self):
        return (1,), (-1,)

    def run(self):
        self.running = True
        print("Generating particle particle pairs ...")
        while self.running:
            alice, bob = self.emit()
            self.socket.send_multipart([b'alice', msgpack.dumps(alice)])
            self.socket.send_multipart([b'bob', msgpack.dumps(bob)])
            time.sleep(0.0)


class Station(object):
    """Detect a particle with a given/random setting"""

    def __init__(self, source: str, arm: str):
        self.arm = arm
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect("tcp://{source}:10001")
        self.socket.setsockopt(zmq.SUBSCRIBE, arm.encode('utf-8'))
        self.running = False

    def save(self, fname):
        """Save the results"""
        f = gzip.open(fname, 'wb')
        numpy.save(f, self.results)
        f.close()

    def detect(self, setting, particle):
        """
        Calculate and return the station outcome
        :param setting: detector setting
        :param particle: particle to detect
        :return: (timestamp, setting, outcome) tuple
        """
        return [time.time(), setting, 1]

    def run(self, settings, seed=None):
        self.running = True
        print(f"Detecting particles for {self.arm}'s arm")
        while self.running:
            src_data = self.socket.recv_multipart()
            setting = random.choice(settings)
            particle = msgpack.loads(src_data[1])
            self.
        pool = mp.Pool()
        results = pool.map(detect_particle, infos)
        print("Done: {0} particles detected in {1:5.1f} sec!".format(len(particles), time.time() - st))
        self.results = numpy.array(results)
        self.save("%s.npy.gz" % self.name)
