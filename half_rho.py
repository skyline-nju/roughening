""" Locate the interface of band by half rho.

    For given y, locate the edge of band at x where rho_x = 1/2 rho_x_max.
"""
import sys
import numpy as np
import platform
import matplotlib
from scipy.ndimage import gaussian_filter

if platform.system() is not "Windows":
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
else:
    import matplotlib.pyplot as plt


def untangle(x, L):
    """ Turn a periodic array into a continous one. """
    x_new = x.copy()
    phase = 0
    for i in range(1, x.size):
        dx = x[i] - x[i - 1]
        if dx > L * 0.5:
            phase -= L
        elif dx < -L * 0.5:
            phase += L
        x_new[i] = x[i] + phase
    return x_new


def find_idx_max(rho_x, pre_idx_max, idx_range=20, mode=0):
    def get_max(i_beg, i_end):
        tmp = np.array([
            rho_x[i - n] if i >= n else rho_x[i] for i in range(i_beg, i_end)
        ])
        return np.argmax(tmp) + i_beg

    n = rho_x.size
    i_beg = pre_idx_max - idx_range
    i_end = pre_idx_max + idx_range
    idx = get_max(i_beg, i_end)
    iter_count = 0
    max_iteration = 3
    if idx == i_end - 1:
        while iter_count < max_iteration:
            i_beg = idx
            i_end = i_beg + idx_range
            idx = get_max(i_beg, i_end)
            if idx < i_end - 1:
                break
            else:
                iter_count += 1
    elif idx == i_beg:
        while iter_count < max_iteration:
            i_end = idx + 1
            i_beg = i_end - idx_range - 1
            idx = get_max(i_beg, i_end)
            if idx > i_beg:
                break
            else:
                iter_count += 1
    if iter_count < max_iteration:
        idx = idx % n
        if mode == 2:
            rho_h = rho_x[idx] * 0.5
            find_count = 0
            idx_h = None
            for i in range(0, n):
                if rho_x[(i - 1) % n] >= rho_h >= rho_x[i % n]:
                    idx_h = i % n
                    find_count += 1
            if find_count != 1:
                idx = None
            return idx, idx_h
        elif mode == 1:
            rho_h = rho_x[idx] * 0.5
            x1 = 0
            for i in range(n // 4 + idx, idx, -1):
                if rho_x[(i - 1) % n] >= rho_h >= rho_x[i % n]:
                    x1 = i % n
            x2 = 0
            for i in range(idx, n // 4 + idx):
                if rho_x[(i - 1) % n] > rho_h >= rho_x[i % n]:
                    x2 = i % n
            if x1 != x2:
                idx = None
            return idx, x1
        else:
            idx_m = np.argmax(rho_x)
            dx = idx - idx_m
            if 0 < dx < n / 2 or dx < -n / 2:
                pass
            else:
                idx = idx_m
            return idx
    else:
        # output error information
        print("Error when finding the optimal index where rho_x gets max.")
        if platform.system() is "Windows":
            plt.plot(rho_x, "-o")
            plt.axvline(pre_idx_max, c="r")
            plt.axvline(pre_idx_max - idx_range, c="b")
            plt.axvline(pre_idx_max + idx_range, c="b")
            plt.axvline(pre_idx_max - 2 * idx_range, c="g")
            plt.axvline(pre_idx_max + 2 * idx_range, c="g")
            plt.show()
        sys.exit()


def find_rho_half(rho_x,
                  idx_max,
                  idx_h=None,
                  xh_pre=None,
                  rho_h_pre=None,
                  debug=False):
    rho_h = 0.5 * rho_x[idx_max]
    n = rho_x.size
    xh = None
    if idx_h is not None:
        rho0 = rho_x[(idx_h - 1) % n]
        rho1 = rho_x[idx_h]
        x0 = idx_h - 0.5
        x1 = idx_h + 0.5
        xh = x0 - (rho0 - rho_h) / (rho0 - rho1) * (x0 - x1)
    else:
        if xh_pre is None and rho_h is None:
            print("Need xh_pre and rho_h.")
            sys.exit()
        gap = xh_pre - (idx_max + 0.5)
        if 0 < gap < n / 2 or gap < -n / 2:
            is_find_xh = False
            for i in range(int(xh_pre) + 6, int(xh_pre) - 5, -1):
                rho0 = rho_x[(i - 1) % n]
                rho1 = rho_x[i % n]
                if rho0 > rho_h >= rho1:
                    x0 = i - 0.5
                    x1 = i + 0.5
                    xh = x0 - (rho0 - rho_h) / (rho0 - rho1) * (x0 - x1)
                    is_find_xh = True
                    break
                idx_end = int(xh_pre) + 30
            if not is_find_xh:
                if idx_end < idx_max:
                    xh_pre += n
                    idx_end += n
                drho = np.array([
                    abs(rho_x[i % n] - rho_h)
                    for i in range(idx_max + 1, idx_end)
                ])
                idx0 = np.argmin(drho) + idx_max + 1
                xh = 0.5 * (idx0 + 0.5 + xh_pre)
                rho_h = 0.5 * (rho_x[idx0 % n] + rho_h_pre)
        else:
            for i in range(idx_max, idx_max + 80):
                rho0 = rho_x[(i - 1) % n]
                rho1 = rho_x[i % n]
                if rho0 > rho_h >= rho1:
                    x0 = i - 0.5
                    x1 = i + 0.5
                    xh = x0 - (rho0 - rho_h) / (rho0 - rho1) * (x0 - x1)
                    break
    if debug:
        import matplotlib.pyplot as plt
        plt.plot(rho_x)
        plt.axhline(rho_h, c="r", linestyle="--")
        plt.axvline(idx_max, c="k")
        plt.axvline(xh, c="r")
        plt.axvline(xh_pre, c="b")
        plt.axhline(rho_h_pre, c="b", linestyle="--")
        plt.show()
        plt.close()

    if xh > n:
        xh -= n

    return xh, rho_h


def find_interface(rho, sigma, debug=False):
    rho_s = gaussian_filter(rho, sigma=sigma, mode="wrap")
    nrows = rho_s.shape[0]
    xh = np.zeros(nrows)
    rho_h = np.zeros(nrows)
    if debug:
        plt.imshow(rho_s.T, origin="lower", interpolation="none")
        plt.show()
        plt.close()
    # find starting index
    start_row = None
    idx_max_pre = np.argmax(np.mean(rho_s, axis=0))
    for row, rhox in enumerate(rho_s):
        idx_max, idx_h = find_idx_max(rhox, idx_max_pre, mode=2)
        if idx_max is not None:
            start_row = row
            idx_max_pre = idx_max
            xh[row], rho_h[row] = find_rho_half(rhox, idx_max, idx_h)
            break
    if idx_max is None:
        for row, rhox in enumerate(rho_s):
            idx_max, idx_h = find_idx_max(rhox, idx_max_pre, mode=1)
            if idx_max is not None:
                start_row = row
                idx_max_pre = idx_max
                xh[row], rho_h[row] = find_rho_half(rhox, idx_max, idx_h)
                break
    if idx_max is None:
        print("Cannot find the starting row.")
        sys.eixt()

    # find xh, rho_h for each row
    for row in range(start_row + 1, start_row + nrows):
        if row >= nrows:
            row -= nrows
        rhox = rho_s[row]
        idx_max = find_idx_max(rhox, idx_max_pre)

        if debug:
            print("row = ", row)
        if debug and row > 1950:
            is_debug = True
        else:
            is_debug = False
        xh[row], rho_h[row] = find_rho_half(
            rhox,
            idx_max,
            xh_pre=xh[row - 1],
            rho_h_pre=rho_h[row - 1],
            debug=is_debug)
        idx_max_pre = idx_max

    if debug:
        mask = rho_s > 1
        plt.imshow(mask.T, origin="lower", interpolation="none")
        yh = np.linspace(0.5, nrows - 0.5, nrows)
        plt.plot(yh, xh, "k")
        plt.show()
        plt.close()
    return xh, rho_h


if __name__ == "__main__":
    import os
    import load_snap
    if platform.system() is not "Windows":
        os.chdir(r"snap_one")
    else:
        os.chdir(r"D:\tmp")
    Lx = int(sys.argv[1])
    Ly = int(sys.argv[2])
    eta = 0.35
    eps = 0
    dt = 2000
    seed = 1234
    snap = load_snap.RawSnap(r"so_%g_%g_%d_%d_%d_%d_%d.bin" %
                             (eta, eps, Lx, Ly, Lx * Ly, dt, seed))
    file = r"wh_%g_%g_%d_%d_%d_%d_%d.dat" % (eta, eps, Lx, Ly, Lx * Ly, dt,
                                             seed)
    f = open(file, "w")
    sigma_y = [5, 10, 15, 20, 25]
    for i, frame in enumerate(snap.gene_frames()):
        x, y, theta = frame
        rho = load_snap.coarse_grain2(
            x, y, theta, Lx=Lx, Ly=Ly, ncols=Lx, nrows=Ly) * 1.0
        line = ""
        for sy in sigma_y:
            try:
                xh, rho_h = find_interface(rho, sigma=[sy, 1])
                w = np.var(untangle(xh, Lx))
            except:
                w = -1
            if len(line) == 0:
                line += "%d\t%f" % (i, w)
            else:
                line += "\t%f" % (w)
        line += "\n"
        f.write(line)
        if i % 100 == 0:
            print(i)
    f.close()
