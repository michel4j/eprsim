#
#    EPR Simulation Framework
#    Copyright (C) 2022  Michel Fodje
#
import datetime
import os
import random
import sys
import time
from abc import ABC, abstractmethod

import msgpack
import numpy
import pandas
import zmq
from tqdm import tqdm

DATA_DTYPE = [
    ('time', 'f8'),
    ('setting', 'f2'),
    ('outcome', 'f2'),
    ('index', 'i4')
]
BUFFER_SIZE = 10

# random number generator
rng = numpy.random.default_rng()


class SourceType(ABC):
    """
    Generate and emit two particles with hidden variables
    """
    RATE = 1e4  # maximum number of particles emitted per second
    JITTER = 0  # Jitter unit in seconds

    def __init__(self):
        self.context = zmq.Context()
        self.alice = self.context.socket(zmq.PUB)
        self.bob = self.context.socket(zmq.PUB)
        for i, socket in enumerate((self.alice, self.bob)):
            socket.setsockopt(zmq.SNDHWM, BUFFER_SIZE)
            socket.bind(f"tcp://*:1000{i + 1}")

        self.running = False
        self.index = 0
        self.clock = 0

    def time(self):
        """
        Return the current emission time. Represents a global clock.
        """
        return float(self.clock + rng.poisson(1) * self.JITTER + self.index * 1 / self.RATE)

    @abstractmethod
    def emit(self):
        """
        Models should implement this. Must return two tuples, corresponding to Alice and Bob's particles
        """
        return (1,), (-1,)

    def stop(self, *args):
        """
        Stop the source
        """
        self.running = False

    def run(self):
        """
        Main loop for emitting particles through the network to listening stations
        """
        self.running = True
        while self.running:
            alice, bob = self.emit()
            a_msg = [msgpack.dumps(self.time()), msgpack.dumps(alice), msgpack.dumps(self.index)]
            b_msg = [msgpack.dumps(self.time()), msgpack.dumps(bob), msgpack.dumps(self.index)]
            # set buffer size after first message
            if self.index == 0:
                size = sys.getsizeof(a_msg) * BUFFER_SIZE
                self.alice.setsockopt(zmq.SNDBUF, size)
                self.bob.setsockopt(zmq.SNDBUF, size)
            self.alice.send_multipart(a_msg)
            self.bob.send_multipart(b_msg)
            self.index += 1
            time.sleep(1 / self.RATE)
        print("Particle Source Stopped!")


class StationType(ABC):
    """
    Detect a particle with a given/random setting
    """

    # maximum frequency of particle emission
    RATE = 1e4

    # Jitter unit in seconds
    JITTER = 0

    # precision of time measurement in number of decimal places,
    # negative for significant digits. used to implement event ready windows
    TIME_PRECISION = 9

    def __init__(self, source: str, arm: str, label: str = ''):
        self.arm = arm
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        port = {'alice': 10001, 'bob': 10002}.get(arm)
        self.socket.connect(f"tcp://{source}:{port}")
        self.socket.setsockopt(zmq.SUBSCRIBE, b'')
        self.socket.setsockopt(zmq.RCVHWM, BUFFER_SIZE)
        self.poller = zmq.Poller()
        self.poller.register(self.socket, zmq.POLLIN)
        self.running = False

        suffix = f'-{label}' if label else ''
        self.filename = f"{self.arm[0].upper()}-{datetime.datetime.now().strftime('%y%m%dT%H')}{suffix}.h5"
        self.clock = 0
        self.index = 0
        if os.path.exists(self.filename):
            os.remove(self.filename)

    @abstractmethod
    def detect(self, setting, particle):
        """
        Calculate and return the station outcome. Models should implement this method
        :param setting: detector setting
        :param particle: particle to detect
        :return: timestamp, setting, outcome tuple, incr is the number of detections
        """
        return self.time(), setting, 1

    def time(self):
        """
        Return time at local synchronized clock. Include poisson jitter and round time to required precision.
        """
        t = float(self.clock + rng.poisson(1) * self.JITTER + self.index * 1 / self.RATE)
        return numpy.round(t, decimals=self.TIME_PRECISION)

    def stop(self, *args):
        """
        Stop detecting particles and close all files
        """
        self.running = False

    def run(self, settings, seed=None):
        """

        :param settings:
        :param seed:
        :return:
        """
        self.running = True
        with pandas.HDFStore(self.filename, complevel=9, complib='blosc:zstd') as store:
            count = 0
            chunk_size = 500
            buffer = numpy.zeros((chunk_size,), dtype=DATA_DTYPE)

            progress = tqdm(total=float("inf"))
            while self.running:
                for i in range(chunk_size):

                    # read particle from source
                    socks = dict(self.poller.poll(1))
                    while self.running and self.socket not in socks:
                        socks = dict(self.poller.poll(1))

                    if not self.running:
                        df = pandas.DataFrame.from_records(buffer[:i])
                        store.append('data', df, format='table', index=False)
                        break

                    elif self.socket in socks:
                        src_data = self.socket.recv_multipart()
                        clock_data, particle_data, index_data = src_data
                        particle = msgpack.loads(particle_data)
                        self.index = msgpack.loads(index_data)
                        self.clock = msgpack.loads(clock_data)

                        # set packet size after first message
                        if count == 0:
                            size = sys.getsizeof(src_data) * BUFFER_SIZE
                            self.socket.setsockopt(zmq.RCVBUF, size)

                        setting = random.choice(settings)
                        results = self.detect(setting, particle)  # detect particle
                        time.sleep(1 / self.RATE)

                        # Record data in chunks to HDF5 file
                        buffer[i] = results + (self.index,)
                        count += 1
                        progress.update(1)

                else:
                    df = pandas.DataFrame.from_records(buffer)
                    store.append('data', df, format='table', index=False)
        print(f"Done: {count} particles detected!")
