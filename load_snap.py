""" Method to read raw or coarse-grained snapshots.

    Raw snapshot record the instant location (x, y) and orientation (theta)
    of each particle with float number (float32).

    Coarse-grained snapshots record the count of number and mean velocity
    over cells with linear size (lx, ly). Types of num, vx, vy are int32,
    float32, float32 for "iff" and unsigned char, signed char, singed char
    for "Bbb".Additional imformation such as time step, sum of vx and vy would
    be also saved in the file.

    FIND A BUG:
    code:
    --------
        f = open(file, "rb")
        buff = f.read(20)
        a = struct.unpack('idd', buff)

    output:
    --------
        struct.error: unpack requires a bytes object of length 24
"""

import numpy as np
import struct
import sys


class Snap:
    """ Base class for snapshot. """

    def __init__(self, file):
        """ Need rewrite for subclass. """
        self.open_file(file)

    def __del__(self):
        self.f.close()

    def open_file(self, file):
        self.f = open(file, "rb")
        self.f.seek(0, 2)
        self.file_size = self.f.tell()
        self.f.seek(0)

    def one_frame(self):
        """ Need rewrite for subclass. """
        pass

    def read_frame(self, idx=0):
        offset = idx * self.frame_size
        if self.file_size - offset >= self.frame_size:
            self.f.seek(offset)
            return self.one_frame()
        else:
            print("Error, index of frame should be less than %d" %
                  (self.file_size // self.frame_size))
            sys.exit()

    def gene_frames(self, beg_idx=0, end_idx=None):
        self.f.seek(beg_idx * self.frame_size)
        if end_idx is None:
            max_size = self.file_size
        else:
            max_size = end_idx * self.frame_size
        while max_size - self.f.tell() >= self.frame_size:
            yield self.one_frame()


class RawSnap(Snap):
    def __init__(self, file):
        self.open_file(file)
        str_list = file.split("_")
        if str_list[0] == "so":
            self.N = int(str_list[5])
        else:
            self.N = self.file_size // 12
        self.frame_size = self.N * 3 * 4
        self.fmt = "%df" % (3 * self.N)

    def one_frame(self):
        buff = self.f.read(self.frame_size)
        data = struct.unpack(self.fmt, buff)
        frame = np.array(data, float).reshape(self.N, 3).T
        return frame


class CoarseGrainSnap(Snap):
    def __init__(self, file):
        str_list = file.split("_")
        self.snap_format = str_list[0].replace("c", "")
        self.ncols = int(str_list[5])
        self.nrows = int(str_list[6])
        self.N = self.ncols * self.nrows
        print(self.ncols, self.nrows)
        if self.snap_format == "Bbb":
            self.fmt = "%dB%db" % (self.N, 2 * self.N)
            self.snap_size = self.N * 3
        elif self.snap_format == "iff":
            self.fmt = "%di%df" % (self.N, 2 * self.N)
            self.snap_size = self.N * 3 * 4
        self.frame_size = self.snap_size + 20
        self.open_file(file)

    def one_frame(self):
        buff = self.f.read(4)
        t, = struct.unpack("i", buff)
        buff = self.f.read(16)
        vxm, vym = struct.unpack("dd", buff)
        buff = self.f.read(self.snap_size)
        data = struct.unpack(self.fmt, buff)
        num = np.array(data[:self.N], int).reshape(self.nrows, self.ncols)
        vx = np.array(data[self.N:2 * self.N], float).reshape(self.nrows,
                                                              self.ncols)
        vy = np.array(data[2 * self.N:3 * self.N], float).reshape(self.nrows,
                                                                  self.ncols)
        if self.snap_format == "Bbb":
            vx /= 128
            vy /= 128
        frame = [t, vxm, vym, num, vx, vy]
        return frame


if __name__ == "__main__":
    """ Just for test. """
    import matplotlib.pyplot as plt
    import os
    # os.chdir(r"D:\code\VM\VM\coarse")
    # file = r"cBbb_0.35_0_140_200_140_200_28000_1.08_13.bin"
    # snap = CoarseGrainSnap(file)
    # frames = snap.gene_frames(90, 100)
    # for i, frame in enumerate(frames):
    #     plt.contourf(frame[4])
    #     plt.title(r"$t=%d$" % frame[0])
    #     plt.show()
    #     plt.close()

    os.chdir(r"D:\tmp")
    file = r'so_0.35_0_150_400_60000_2000_1234.bin'
    snap = RawSnap(file)
    for i, frame in enumerate(snap.gene_frames(100, 110)):
        x, y, theta = frame
        plt.plot(x, y, "o")
        plt.show()
        plt.close()
