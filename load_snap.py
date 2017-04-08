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
import sys
import struct
import numpy as np
import matplotlib.pyplot as plt


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
        self.Lx = int(str_list[3])
        self.Ly = int(str_list[4])
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


def coarse_grain(x, y, theta=None, Lx=None, Ly=None, ncols=None, nrows=None):
    """ Coarse grain the raw snapshot over boxes with size (lx, ly)

        Parameters:
        --------
        x, y: np.ndarray
            Coordination of particles.
        theta: np.ndarray, optional
            Moving direction of particles, return coarse grain velocity field
            if given.
        ncols, nrows: int, optional
            Number of columns and rows of boxes for coarse grain.

        Returns:
        --------
        rho: np.ndarray
            Coarse grained density field.
        vx, vy: np.ndarray
            Coarse grained velocity fields.
    """

    if Lx is None:
        Lx = int(np.round(x.max()))
    if Ly is None:
        Ly = int(np.round(y.max()))
    if ncols is None:
        ncols = Lx
    if nrows is None:
        nrows = Ly
    print(Lx, Ly, ncols, nrows)
    lx = Lx / ncols
    ly = Ly / nrows

    n = x.size
    rho = np.zeros((nrows, ncols), int)
    if theta is not None:
        flag_v = True
        vx = np.zeros((nrows, ncols))
        vy = np.zeros((nrows, ncols))
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
    else:
        flag_v = False

    for i in range(n):
        col = int(x[i]/lx)
        row = int(y[i]/ly)
        rho[row, col] += 1
        if flag_v:
            vx[row, col] += cos_theta[i]
            vy[row, col] += sin_theta[i]
    mask = rho > 0
    vx[mask] /= rho[mask]
    vy[mask] /= rho[mask]
    rho = rho / (lx * ly)
    vx /= (lx * ly)
    vy /= (lx * ly)
    return rho, vx, vy


def plot_coarse_grained_snap(rho, vx, vy=None, ax=None, t=None):
    plt.subplot(121)
    rho[rho > 4] = 4
    plt.contour(rho, vmax=4, level=[0, 1, 2, 3, 4], extend="max")
    plt.colorbar()
    plt.subplot(122)
    plt.contourf(vx)
    plt.colorbar()
    if t is not None:
        plt.suptitle(r"$t=%d$" % t)
    plt.show()
    plt.close()


if __name__ == "__main__":
    """ Just for test. """
    import os
    # os.chdir(r"D:\code\VM\VM\coarse")
    # file = r"cBbb_0.35_0_140_200_140_200_28000_1.08_13.bin"
    # snap = CoarseGrainSnap(file)
    # frames = snap.gene_frames(90, 100)
    # for i, frame in enumerate(frames):
    #     rho = frame[3]
    #     vx = frame[4]
    #     plot_coarse_grained_snap(rho, vx, frame[0])

    os.chdir(r"D:\tmp")
    file = r'so_0.35_0_150_900_135000_2000_1234.bin'
    snap = RawSnap(file)
    for i, frame in enumerate(snap.gene_frames(200, 202)):
        x, y, theta = frame
        rho, vx, vy = coarse_grain(x, y, theta, nrows=100)
        plot_coarse_grained_snap(rho, vx, t=i)
