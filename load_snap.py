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

    def gene_frames(self, beg_idx=0, end_idx=None, interval=1):
        self.f.seek(beg_idx * self.frame_size)
        if end_idx is None:
            max_size = self.file_size
        else:
            max_size = end_idx * self.frame_size
        count = 0
        while max_size - self.f.tell() >= self.frame_size:
            if count % interval == 0:
                yield self.one_frame()
            else:
                self.f.seek(self.frame_size, 1)
            count += 1


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

    def show(self, beg_idx=0, end_idx=None, interval=1, markersize=1):
        for i, frame in enumerate(
                self.gene_frames(beg_idx, end_idx, interval)):
            x, y, theta = frame
            plt.plot(x, y, "o", ms=markersize)
            plt.title("frame %d" % (interval * i))
            plt.show()
            plt.close()


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


def coarse_grain(x,
                 y,
                 theta=None,
                 Lx=None,
                 Ly=None,
                 ncols=None,
                 nrows=None,
                 norm=True):
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
    ncols_over_Lx = ncols / Lx
    nrows_over_Ly = nrows / Ly

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
        col = int(x[i] * ncols_over_Lx)
        row = int(y[i] * nrows_over_Ly)
        rho[row, col] += 1
        if flag_v:
            vx[row, col] += cos_theta[i]
            vy[row, col] += sin_theta[i]
    cell_area = Lx / ncols * Ly / nrows
    if flag_v:
        mask = rho > 0
        vx[mask] /= rho[mask]
        vy[mask] /= rho[mask]
        if norm:
            rho = rho / cell_area
            vx /= cell_area
            vy /= cell_area
        return rho, vx, vy
    else:
        if norm:
            rho = rho / cell_area
        return rho


def coarse_grain2(x, y, theta, Lx=None, Ly=None, ncols=None, nrows=None):
    if Lx is None:
        Lx = int(np.round(x.max()))
    if Ly is None:
        Ly = int(np.round(y.max()))
    if ncols is None:
        ncols = Lx
    if nrows is None:
        nrows = Ly
    ncols_over_Lx = ncols / Lx
    nrows_over_Ly = nrows / Ly
    vx = np.cos(theta)

    num = np.zeros((nrows, ncols), int)
    n = x.size
    for i in range(n):
        col = int(x[i] * ncols_over_Lx)
        if col >= ncols:
            col = ncols-1
        row = int(y[i] * nrows_over_Ly)
        if row >= nrows:
            row = nrows - 1
        if (vx[i] > 0):
            num[row, col] += 1
    return num


def plot_contour(rho, vx, vy=None, ax=None, t=None):
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
    os.chdir(r"D:\tmp")
    Lx = 150
    Ly = 50
    N = Lx * Ly
    dt = 50000
    seed = 1
    file = r'so_0.35_0_%d_%d_%d_%d_%d.bin' % (Lx, Ly, N, dt, seed)
    snap = RawSnap(file)
    snap.show(interval=1)
