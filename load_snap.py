""" Method to read raw or coarse-grained snapshots.

    Raw snapshot record the instant location (x, y) and orientation (theta)
    of each particle with float number (float32).

    Coarse-grained snapshots record the count of number and mean velocity
    over cells with linear size (lx, ly). Types of num, vx, vy are int32,
    float32, float32 for "iff" and unsigned char, signed char, singed char
    for "Bbb". Additional imformation such as time step, sum of vx and vy would
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
import os
import sys
import glob
import struct
import numpy as np
import platform
import matplotlib
from scipy.ndimage import gaussian_filter

if platform.system() is not "Windows":
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
else:
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

    def get_num_frames(self):
        return self.file_size // self.frame_size


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
        self.file = file
        print(self.ncols, self.nrows)
        if self.snap_format == "Bbb":
            self.fmt = "%dB%db" % (self.N, 2 * self.N)
            self.snap_size = self.N * 3
        elif self.snap_format == "iff":
            self.fmt = "%di%df" % (self.N, 2 * self.N)
            self.snap_size = self.N * 3 * 4
        elif self.snap_format == "B":
            self.fmt = "%dB" % (self.N)
            self.snap_size = self.N
        self.frame_size = self.snap_size + 20
        self.open_file(file)

    def one_frame(self):
        buff = self.f.read(4)
        t, = struct.unpack("i", buff)
        buff = self.f.read(16)
        vxm, vym = struct.unpack("dd", buff)
        buff = self.f.read(self.snap_size)
        data = struct.unpack(self.fmt, buff)
        if self.snap_format == "B":
            num = np.array(data, int).reshape(self.nrows, self.ncols)
            frame = [t, vxm, vym, num]
        else:
            num = np.array(data[:self.N], int).reshape(self.nrows, self.ncols)
            vx = np.array(data[self.N:2 * self.N], float).reshape(self.nrows,
                                                                  self.ncols)
            vy = np.array(data[2 * self.N:3 * self.N], float).reshape(
                self.nrows, self.ncols)
            if self.snap_format == "Bbb":
                vx /= 128
                vy /= 128
            frame = [t, vxm, vym, num, vx, vy]
        return frame

    def show(self,
             i_beg=0,
             i_end=None,
             di=1,
             lx=1,
             ly=1,
             transpos=True,
             sigma=[5, 1],
             output=True,
             show=True):
        import half_rho
        import half_peak
        if output:
            f = open(self.file.replace(".bin", "_%d.dat" % (sigma[0])), "w")
        for i, frame in enumerate(self.gene_frames(i_beg, i_end, di)):
            t, vxm, vym, num = frame
            rho = num.astype(np.float32)
            yh = np.linspace(0.5, self.nrows - 0.5, self.nrows)
            # xh1, rho_h1 = half_rho.find_interface(rho, sigma=sigma)
            xh2, rho_h2 = half_peak.find_interface(rho, sigma=sigma)
            # xh1 = half_rho.untangle(xh1, self.ncols)
            xh2 = half_rho.untangle(xh2, self.ncols)
            # w1 = np.var(xh1)
            w2 = np.var(xh2)
            if show:
                if ly > 1:
                    rho = np.array([
                        np.mean(num[i * ly:(i + 1) * ly], axis=0)
                        for i in range(self.nrows // ly)
                    ])
                if lx > 1:
                    rho = np.array([
                        np.mean(rho[:, i * lx:(i + 1) * lx], axis=1)
                        for i in range(self.ncols // lx)
                    ])
                if transpos:
                    rho = rho.T
                    box = [0, self.nrows, 0, self.ncols]
                    plt.figure(figsize=(14, 3))
                    plt.plot(yh, xh2, "r")
                    plt.xlabel(r"$y$")
                    plt.ylabel(r"$x$")
                else:
                    box = [0, self.ncols, 0, self.nrows]
                    plt.figure(figsize=(4, 12))
                    plt.plot(xh2, yh, "r")
                    plt.xlabel(r"$x$")
                    plt.ylabel(r"$y$")
                rho = gaussian_filter(rho, sigma=sigma)
                rho[rho > 4] = 4
                plt.imshow(
                    rho,
                    origin="lower",
                    interpolation="none",
                    extent=box,
                    aspect="auto")
                plt.title(r"$t=%d, \phi=%g, w^2=%g$" %
                          (t, np.sqrt(vxm**2 + vym**2), w2))
                plt.tight_layout()
                plt.show()
                plt.close()
            print(t, np.sqrt(vxm**2 + vym**2), w2)
            if output:
                f.write("%d\t%f\t%f\n" %
                        (t, np.sqrt(vxm**2 + vym**2), w2))
        if output:
            f.close()


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
        if vx[i] > 0:
            col = int(x[i] * ncols_over_Lx)
            if col >= ncols:
                col = ncols - 1
            row = int(y[i] * nrows_over_Ly)
            if row >= nrows:
                row = nrows - 1
            num[row, col] += 1
    return num


def show_separated_snaps(Lx,
                         Ly,
                         seed,
                         t_beg,
                         t_end,
                         dt=None,
                         eta=0.35,
                         eps=0,
                         rho0=1,
                         transpos=False,
                         sigma=[5, 1]):
    from half_peak import find_interface
    from half_rho import untangle
    if dt is None:
        dt = t_beg
    t = t_beg
    while t <= t_end:
        file = r's_%g_%g_%g_%d_%d_%d_%08d.bin' % (eta, eps, rho0, Lx, Ly, seed,
                                                  t)
        snap = RawSnap(file)
        for frame in snap.gene_frames():
            try:
                x, y, theta = frame
                phi = np.sqrt(
                    np.mean(np.cos(theta))**2 + np.mean(np.sin(theta))**2)
                rho = coarse_grain2(x, y, theta, Lx=Lx, Ly=Ly).astype(float)
                yh = np.linspace(0.5, Ly - 0.5, Ly)
                xh, rho_h = find_interface(rho, sigma=sigma)
                w = np.var(untangle(xh, Lx))
                print("t=%d, phi=%f, w=%f" % (t, phi, w))
                plt.subplot(121)
                if transpos:
                    x, y = y, x
                    xh, yh = yh, xh
                    plt.xlim(0, Ly)
                    plt.ylim(0, Lx)
                else:
                    plt.xlim(0, Lx)
                    plt.ylim(0, Ly)
                plt.scatter(x, y, s=1, c=theta, cmap="hsv")
                plt.plot(xh, yh)
                plt.title(r"$t=%d, \phi=%g, w^2=%g$" % (t, phi, w))
                plt.colorbar()
                plt.subplot(122)
                plt.plot(rho.mean(axis=0))
                plt.show()
                plt.close()
            except:
                print("t=%d, Error" % t)
        t += dt


def handle_file():
    os.chdir("coarse")
    files = glob.glob("cB*.bin")
    for file in files:
        try:
            snap = CoarseGrainSnap(file)
            snap.show(show=False)
        except:
            print("error when handle ", file)


def handle_raw_snap(file, sigma_y):
    import load_snap
    import half_peak
    import snake
    from half_rho import untangle
    str_list = file.split("_")
    Lx = int(str_list[3])
    Ly = int(str_list[4])
    sigma = [sigma_y, 1]
    snap = load_snap.RawSnap(file)
    n = snap.get_num_frames()
    h1 = np.zeros((n, Ly), np.float32)
    h2 = np.zeros((n, Ly), np.float32)
    for i, frame in enumerate(snap.gene_frames()):
        x, y, theta = frame
        rho = load_snap.coarse_grain2(
            x, y, theta, Lx=Lx, Ly=Ly, ncols=Lx, nrows=Ly).astype(float)
        xh1, rho_h = half_peak.find_interface(rho, sigma=sigma)
        rho_s = gaussian_filter(rho, sigma=sigma)
        xh2, yh2 = snake.find_interface(
            rho_s, 0.5, 0.1, 0.25, 400, rho_h, dx=5)
        h1[i] = untangle(xh1, Lx)
        h2[i] = untangle(xh2, Lx)
        print("i=", i)
    outfile = file.replace(".bin", "_%d.npz" % sigma_y)
    np.savez(outfile, h1=h1, h2=h2)


if __name__ == "__main__":
    """ Just for test. """
    # os.chdir(r"D:\code\VM\VM\snap")
    # show_separated_snaps(
    #     180,
    #     20,
    #     312,
    #     1000,
    #     100000,
    #     1000,
    #     0.35,
    #     0,
    #     transpos=True,
    #     sigma=[5, 1])
    # snap = RawSnap(r"s_0.35_0_1_180_20_313_00800000.bin")
    # snap.show()
    os.chdir(r"D:\tmp")
    snap = CoarseGrainSnap(
        "cB_0.35_0_180_25600_180_25600_4608000_1.06_4232.bin")
    snap.show(sigma=[15, 1], show=False)
