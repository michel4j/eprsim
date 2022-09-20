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
import hdf5plugin

import datetime


class SourceType(ABC):
    """
    Generate and emit two particles with hidden variables
    """

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
            time.sleep(1e-3)
        print("Particle Source Stopped!")


class StationType(ABC):
    """
    Detect a particle with a given/random setting
    """

    def __init__(self, source: str, arm: str):
        self.arm = arm
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect(f"tcp://{source}:10001")
        self.socket.setsockopt(zmq.SUBSCRIBE, arm.encode('utf-8'))
        self.running = False
        self.filename = f"{self.arm}-{datetime.datetime.now().strftime('%Y%m%dT%H%M%S')}.h5"
        self.sync_time = datetime.datetime.combine(datetime.datetime.today(), datetime.time.min).timestamp()

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
        return datetime.datetime.now().timestamp() - self.sync_time

    def stop(self):
        """
        Stop detecting particles and close all files
        """
        self.running = False

    def run(self, settings, seed=None):
        self.running = True

        start_time = time.time()
        count = 0

        with h5py.File(self.filename, "a") as fobj:
            chunk_size = 1**6
            container_size = 0
            dset = fobj.create_dataset('data', (chunk_size, 3), maxshape=(None, 3), dtype='<f8', chunks=True, **hdf5plugin.Zstd())
            buffer = numpy.zeros((chunk_size, 3))

            print(f"Detecting particles for {self.arm}'s arm")
            while self.running:
                # read particle from source
                src_data = self.socket.recv_multipart()
                particle = msgpack.loads(src_data[1])

                # select setting randomly
                setting = random.choice(settings)

                # detect particle
                incr, results = self.detect(setting, particle)

                # Record data in chunks to HDF5 file
                i = count % chunk_size
                buffer[i] = results
                if i == (chunk_size - 1) and incr > 0:
                    dset.resize((count,3))
                    dset[container_size:] = buffer[:]
                    container_size = count
                count += int(incr)

            # Add remaining data
            if count > container_size:
                dset.resize(count, axis=0)
                dset[container_size:count] = buffer[:count-container_size]

        elapsed = time.time() - start_time
        print(f"Done: {count} particles detected in {elapsed:5.1f} sec!")


