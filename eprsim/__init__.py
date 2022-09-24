#
#    EPR Simulation Framework
#    Copyright (C) 2022  Michel Fodje
#
from abc import ABC, abstractmethod
import random
import time
import zmq
import msgpack
import numpy
import h5py
from tqdm import tqdm

import datetime

class SourceType(ABC):
    """
    Generate and emit two particles with hidden variables
    """
    EMISSION_TIME = 1e-4

    def __init__(self, port=10001):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind("tcp://*:%s" % port)
        self.running = False

    @abstractmethod
    def emit(self):
        """
        Models should implement this. Must return two tuples, corresponding to Alice and Bob's particles
        """
        return (1,), (-1,)

    def stop(self):
        """
        Stop the source
        """
        self.running = False

    def run(self):
        """
        Main loop for emitting particles through the network to listening stations
        """
        self.running = True
        print("Generating particle particle pairs ...")
        while self.running:
            alice, bob = self.emit()
            self.socket.send_multipart([b'alice', msgpack.dumps(alice)])
            self.socket.send_multipart([b'bob', msgpack.dumps(bob)])
            time.sleep(self.EMISSION_TIME)
        print("Particle Source Stopped!")


class StationType(ABC):
    """
    Detect a particle with a given/random setting
    """
    DETECTION_TIME = 1e-3

    def __init__(self, source: str, arm: str):
        self.arm = arm
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect(f"tcp://{source}:10001")
        self.socket.setsockopt(zmq.SUBSCRIBE, arm.encode('utf-8'))
        self.running = False
        self.filename = f"{self.arm}-{datetime.datetime.now().strftime('%y%m%dT%H%M')}.h5"

    @abstractmethod
    def detect(self, setting, particle):
        """
        Calculate and return the station outcome. Models should implement this method
        :param setting: detector setting
        :param particle: particle to detect
        :return: incr, (timestamp, setting, outcome) tuple, incr is the number of detections
        """
        return 1, (self.time(), setting, 1)

    def time(self):
        """"
        Synchronized clock
        """
        return datetime.datetime.now().timestamp()

    def stop(self):
        """
        Stop detecting particles and close all files
        """
        self.running = False

    def run(self, settings, seed=None):
        self.running = True
        count = 0

        with h5py.File(self.filename, "a") as fobj:
            chunk_size = 10000
            container_size = 0
            buffer = numpy.zeros((chunk_size, 3))
            dset = fobj.create_dataset('data', (chunk_size, 3), maxshape=(None, 3), dtype='<f8', chunks=True, compression='lzf')

            print(f"Detecting particles for {self.arm}'s arm")
            progress = tqdm(total=float("inf"))

            while self.running:
                i = 0
                while i < chunk_size:
                    # read particle from source
                    src_data = self.socket.recv_multipart()
                    particle = msgpack.loads(src_data[1])
                    setting = random.choice(settings)

                    results = self.detect(setting, particle)   # detect particle
                    time.sleep(self.DETECTION_TIME)

                    if results:
                        # Record data in chunks to HDF5 file
                        buffer[i]= results
                        count += 1
                        i += 1
                        progress.update(1)

                    if not self.running:
                        dset.resize(count, axis=0)
                        dset[container_size:count] = buffer[:count - container_size]
                        break

                else:
                    dset.resize(count, axis=0)
                    dset[container_size:] = buffer[:]
                    container_size = count

        print(f"Done: {count} particles detected!")


