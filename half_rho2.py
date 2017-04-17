""" Detect the interface of band.

    Coarse grain the density field with gaussian fileter. For each y, calculate
    x_h at which rho_x is equal to half of the leftmost peak.
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


def find_ini_row(rho, debug=0, rho_h_thresh=2):
    valid_row = []
    valid_dx = []
    valid_idx_h = []
    possible_row = []
    for row, rho_x in enumerate(rho):
        rho_h = 0.5 * rho_x.max()
        if rho_h >= rho_h_thresh:
            count = 0
            idx_h = None
            for i in range(rho_x.size):
                if rho_x[i - 1] >= rho_h >= rho_x[i]:
                    count += 1
                    idx_h = i
            if count == 1:
                idx_m = np.argmax(rho_x)
                if idx_h < idx_m:
                    idx_h += rho_x.size
                is_decreasing = True
                i = (idx_m + 1) % rho_x.size
                while rho_x[i] > 0.5 * rho_h:
                    if rho_x[i] - rho_x[i - 1] > 0:
                        is_decreasing = False
                        break
                    else:
                        i += 1
                        if i >= rho_x.size:
                            i -= rho_x.size
                if is_decreasing:
                    valid_row.append(row)
                    valid_idx_h.append(idx_h % rho_x.size)
                    if idx_h > idx_m:
                        valid_dx.append(idx_h - idx_m)
                    else:
                        valid_dx.append(idx_h + rho_x.size - idx_m)
                else:
                    possible_row.append(row)
    if len(valid_row) == 0:
        for row in possible_row:
            rho_x = rho[row]
            idx_m = np.argmax(rho_x)
            i = idx_m + 1
            iter_count = 0
            idx_peak = None
            while iter_count < 50:
                if rho_x[(i-1) % rho_x.size] > rho_x[i % rho_x.size] and \
                        rho_x[(i-1) % rho_x.size] > rho_x[(i+1) % rho_x.size]:
                    idx_peak = i - 1
                    break
            if idx_peak is not None:
                rho_h = rho_x[idx_peak]
                is_decreasing = True
                while rho_x[i] > 0.5 * rho_h:
                    if rho_x[i] - rho_x[i - 1] > 0:
                        is_decreasing = False
                        break
                    else:
                        i += 1
                        i = i % rho_x.size
                    if is_decreasing:
                        valid_row.append(row)
                        valid_idx_h.append(idx_h % rho_x.size)
                        if idx_h > idx_m:
                            valid_dx.append(idx_h - idx_m)
                        else:
                            valid_dx.append(idx_h + rho_x.size - idx_m)
    if len(valid_dx) > 0:
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


def cal_dis(i, j, n):
    dx = i - j
    if dx > 0.5 * n:
        dx -= n
    elif dx < -0.5 * n:
        dx += n
    return dx


def detect_peak(y,
                i0,
                max_step=None,
                leftward=True,
                y_thresh=None,
                dy_thresh=0.5):
    n = y.size
    peak = []
    vally = []
    if leftward:
        i = i0
        di = -1
    else:
        i = (i0 + 2) % n
        di = 1
    count = 0
    while True:
        if y[i - 1] > y[i] and y[i - 1] > y[i - 2]:
            peak.append(i - 1)
            if len(peak) == 2 and len(vally) >= 1:
                if y[peak[0]] - y[vally[-1]] > y[peak[1]] - y[vally[-1]]:
                    del peak[1]
                else:
                    del peak[0]
                del vally[-1]
        elif y[i - 1] < y[i] and y[i - 1] < y[i - 2]:
            vally.append(i - 1)
            if len(peak) == 1:
                if leftward:
                    dy = y[peak[0]] - y[vally[-1]]
                    dx = cal_dis(peak[0], vally[-1], n)
                    if dx >= 4 or dy >= dy_thresh:
                        break
                else:
                    dx1 = cal_dis(vally[-1], peak[0], n)
                    if dx1 >= 4 and y[peak[0]] >= y_thresh:
                        if len(vally) == 2:
                            dy = y[peak[0]] - y[vally[0]]
                            dx2 = cal_dis(peak[0], vally[0], n)
                            if y[vally[0]] > 0.5 * y_thresh and \
                                    (dx2 >= 4 or dy > dy_thresh):
                                break
                            elif y[vally[0]] > 0.75 * y_thresh and \
                                    (dx2 >= 4 or dy > 0.5 * dy_thresh):
                                break
                        else:
                            break
            elif len(peak) > 1:
                print("Error when detect peak")
                sys.exit()
        i += di
        i = i % n
        count += 1
        if max_step is not None and count > max_step:
            return None
    return peak[0]


def get_next_idx_h(rho_x, idx_h_pre, debug=False):
    n = rho_x.size
    drho_thresh = 0.5
    idx_peak = detect_peak(rho_x, idx_h_pre, dy_thresh=drho_thresh)
    rho_h = rho_x[idx_peak] * 0.5
    idx_peak2 = detect_peak(
        rho_x,
        idx_h_pre,
        y_thresh=rho_h,
        dy_thresh=drho_thresh,
        max_step=40,
        leftward=False)
    if idx_peak2 is not None:
        idx_peak = idx_peak2
        rho_h = rho_x[idx_peak] * 0.5
    i = (idx_peak + 1) % n
    while True:
        if rho_x[i] <= rho_h <= rho_x[i - 1]:
            idx_h = i
            break
        else:
            i += 1
            i = i % n
    if debug:
        plt.plot(rho_x)
        plt.axvline(idx_peak, c="g")
        plt.axvline(idx_h, c="r")
        plt.axvline(idx_h_pre, c="b", linestyle="--")
        plt.axhline(rho_h, c="c")
        plt.show()
        plt.close()
    return idx_h, rho_h


def iteration(rho, row0, idx_h0, left=True, debug=0):
    nrows = rho.shape[0]
    idx_h = np.zeros(nrows, int)
    rho_h = np.zeros(nrows)
    idx_h[row0] = idx_h0
    rho_h[row0] = rho[row0, idx_h0]
    if left:
        row_range = range(row0 + 1, row0 + nrows)
        drow = 1
    else:
        row_range = range(row0 - 1, row0 - nrows, -1)
        drow = -1
    for row in row_range:
        row = row % nrows
        if debug and row > 1000:
            flag = True
        else:
            flag = False
        idx_h[row], rho_h[row] = get_next_idx_h(
            rho[row], idx_h[(row - drow) % nrows], debug=flag)
        if debug:
            print("row = ", row)
    return idx_h, rho_h


def get_xh(rho, idx_h, rho_h):
    xh = np.zeros_like(rho_h)
    for row, rhox in enumerate(rho):
        i = idx_h[row]
        rhoh = rho_h[row]
        rho1 = rhox[i - 1]
        rho2 = rhox[i]
        xh[row] = i - 0.5 + (rho1 - rhoh) / (rho1 - rho2)
    return xh


def find_interface(rho, sigma, debug=0, leftward=False):
    # smooth the density filed
    rho_s = gaussian_filter(rho, sigma=sigma, mode="wrap")
    start_row, start_idx_h = find_ini_row(rho_s, debug=debug)
    idx_h, rho_h = iteration(
        rho_s, start_row, start_idx_h, left=leftward, debug=debug)
    xh = get_xh(rho_s, idx_h, rho_h)
    return xh, rho_h


if __name__ == "__main__":
    import os
    import load_snap
    import half_rho
    os.chdir(r"D:\tmp")
    Lx = 150
    Ly = 250
    snap = load_snap.RawSnap(r"so_%g_%g_%d_%d_%d_%d_%d.bin" %
                             (0.35, 0, Lx, Ly, Lx * Ly, 2000, 1234))
    debug = 0
    t_beg = 2000
    t_end = 2500
    w1 = []
    w2 = []
    for i, frame in enumerate(snap.gene_frames(t_beg, t_end)):
        x, y, theta = frame
        rho = load_snap.coarse_grain2(x, y, theta, Lx=Lx, Ly=Ly).astype(float)
        xh, rho_h = find_interface(rho, sigma=[10, 1], debug=debug)
        xh1, rho_h1 = find_interface(rho, sigma=[10, 1], leftward=True)
        xh2, rho_h2 = half_rho.find_interface(rho, sigma=[10, 1])
        yh = np.linspace(0, Ly - 1, Ly)
        print(i)
        if debug > 0:
            rho[rho > 4] = 4
            plt.imshow(rho.T, interpolation="none", origin="lower")
            plt.plot(yh, xh, "k")
            plt.plot(yh, xh1, "w--")
            plt.plot(yh, xh2, "r")
            plt.show()
            plt.close()
        w1.append(np.var(half_rho.untangle(xh, Lx)))
        w2.append(np.var(half_rho.untangle(xh2, Lx)))
    plt.plot(w1)
    plt.plot(w2)
    plt.show()
