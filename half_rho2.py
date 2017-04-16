""" Detect the interface of band.

    With coarse-grained density fieled rho(y, x), for fixed y, find the first
    peak of rho_x, then calculate x_h where rho_x is half of peak.
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


def find_ini_row(rho, debug=0):
    valid_row = []
    valid_dx = []
    valid_idx_h = []
    is_larger_than_2 = False
    for row, rho_x in enumerate(rho):
        rho_h = 0.5 * rho_x.max()
        if rho_h >= 2:
            is_larger_than_2 = True
            count = 0
            idx_h = None
            for i in range(rho_x.size):
                if rho_x[i-1] >= rho_h >= rho_x[i]:
                    count += 1
                    idx_h = i
            if count == 1:
                idx_m = np.argmax(rho_x)
                if idx_h < idx_m:
                    idx_h += rho_x.size
                is_decreasing = True
                i = (idx_m + 1) % rho_x.size
                while rho_x[i] > 0.5 * rho_h:
                    if rho_x[i] - rho_x[i-1] > 0:
                        is_decreasing = False
                        break
                    else:
                        i += 1
                        if i >= rho_x.size:
                            i -= rho_x.size
                if is_decreasing:
                    valid_row.append(row)
                    valid_idx_h.append(idx_h)
                    if idx_h > idx_m:
                        valid_dx.append(idx_h - idx_m)
                    else:
                        valid_dx.append(idx_h + rho_x.size - idx_m)
    if len(valid_row) == 0:
        print("Failed to find starting row.")
        if not is_larger_than_2:
            print("rho_h=2 may be too large")
        sys.exit()
    idx_min = np.array(valid_dx).argmin()
    start_row = valid_row[idx_min]
    start_idx_h = valid_idx_h[idx_min]
    if debug > 0:
        plt.plot(rho[start_row])
        plt.axvline(start_idx_h)
        plt.title("start row = %d" % start_row)
        plt.show()
        plt.close()
    return start_row, start_idx_h


def get_next_idx_h(rho_x, idx_h_pre, debug=0):
    # find peak behind idx_h_pre
    i = idx_h_pre
    idx_peak = None
    while True:
        if rho_x[i] < rho_x[i-1] and rho_x[i-1] > rho_x[i-2] > rho_x[i-3]:
            idx_peak = i-1
            break
        else:
            i -= 1
    rho_h = rho_x[idx_peak] * 0.5
    i = (idx_peak + 1) % rho_x.size
    while True:
        if rho_x[i] <= rho_h <= rho_x[i-1]:
            idx_h = i
            break
        else:
            i += 1
            if i >= rho_x.size:
                i -= rho_x.size
    return idx_h, rho_h


def find_interface(rho, sigma, debug=0):
    # smooth the density filed
    rho_s = gaussian_filter(rho, sigma=sigma, mode="wrap")
    nrows = rho_s.shape[0]
    xh = np.zeros(nrows, int)
    yh = np.linspace(0.5, nrows - 0.5, nrows)
    rho_h = np.zeros(nrows)
    start_row, start_idx_h = find_ini_row(rho_s, debug)
    xh[start_row] = start_idx_h
    rho_h[start_row] = rho_s[start_row, start_idx_h]
    for row in range(start_row + 1, start_row + nrows):
        if row >= nrows:
            row -= nrows
        xh[row], rho_h[row] = get_next_idx_h(rho_s[row], xh[row-1])
    if debug > 0:
        plt.imshow(rho_s.T, interpolation="none", origin="lower")
        plt.plot(yh, xh)
        plt.show()
        plt.close()
    return xh, yh, rho_h


if __name__ == "__main__":
    import os
    import load_snap
    os.chdir(r"D:\tmp")
    Lx = 150
    Ly = 250
    snap = load_snap.RawSnap(r"so_%g_%g_%d_%d_%d_%d_%d.bin" %
                             (0.35, 0, Lx, Ly, Lx * Ly, 2000, 1234))
    debug = 2
    t_beg = 97
    t_end = 98
    for i, frame in enumerate(snap.gene_frames(t_beg, t_end)):
        x, y, theta = frame
        rho = load_snap.coarse_grain2(x, y, theta, Lx=Lx, Ly=Ly).astype(float)
        xh, yh, rho_h = find_interface(rho, sigma=[5, 1], debug=debug)
