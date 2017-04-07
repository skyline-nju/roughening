""" Method to read raw or coarse-grained snapshots.

    Raw snapshot record the instant location (x, y) and orientation (theta)
    of each particle with float number (float32).

    Coarse-grained snapshots record the count of number and mean velocity
    over cells with linear size (lx, ly). Types of num, vx, vy are int32,
    float32, float32 for "iff" and unsigned char, signed char, singed char
    for "Bbb".

    Additional imformation such as time step, sum of vx and vy would be also
    saved in the file.

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
import os
import struct


def read_coarse_grain(eta, eps, Lx, Ly, ncols, nrows, N, dt, seed, ff):
    """ Read coarse-grained snapshot. """

    file = "c%s_%g_%g_%d_%d_%d_%d_%d_%g_%d.bin" % (ff, eta, eps, Lx, Ly, ncols,
                                                   nrows, N, dt, seed)
    ncell = ncols * nrows
    if ff == "Bbb":
        fmt = "%dB%db" % (ncell, 2 * ncell)
        block_size = ncell * 3
    elif ff == "iff":
        fmt = "%di%df" % (ncell, 2 * ncell)
        block_size = ncell * 3 * 4
    with open(file, "rb") as f:
        while True:
            block = f.read(4)
            if not block:
                break
            else:
                t, = struct.unpack("i", block)
                block = f.read(16)
                vxm, vym = struct.unpack("dd", block)
                block = f.read(block_size)
                buff = struct.unpack(fmt, block)
                num = np.array(buff[:ncell]).reshape(nrows, ncols)
                vx = np.array(buff[ncell:2 * ncell], float).reshape(nrows,
                                                                    ncols)
                vy = np.array(buff[2 * ncell:3 * ncell], float).reshape(nrows,
                                                                        ncols)

                if ff == "Bbb":
                    vx /= 128
                    vy /= 128
                frame = [t, vxm, vym, num, vx, vy]
                yield frame


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    os.chdir(r"D:\code\VM\VM\coarse")
    f = read_coarse_grain(0.35, 0, 140, 200, 140, 200, 28000, 1.08, 13, "Bbb")
    ts = []
    phi = []
    for block in f:
        t, vxm, vym, num, vx, vy = block
        ts.append(t)
        phi.append(np.sqrt(vxm**2 + vym**2))
    plt.plot(ts, phi, "-o")
    plt.xscale("log")
    plt.show()
    plt.close()