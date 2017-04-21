""" Detect the interface of band.

    Coarse grain the density field with gaussian filter. For each y, calculate
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


def cal_dis(i, j, n):
    """ Cal Distance between i, j under the periodic boundary condition. """
    dx = i - j
    if dx > 0.5 * n:
        dx -= n
    elif dx < -0.5 * n:
        dx += n
    return dx


def get_xh(rho, idx_h, rho_h):
    """ Cal x where rho=rho_h by linear interpolation. """
    xh = np.zeros_like(rho_h)
    for row, rhox in enumerate(rho):
        i = idx_h[row]
        rhoh = rho_h[row]
        rho1 = rhox[i - 1]
        rho2 = rhox[i]
        xh[row] = i - 0.5 + (rho1 - rhoh) / (rho1 - rho2)
    return xh


def get_idx_nearest(y, i_beg, y0):
    """ Get the index i0 (>i_beg) where y[i0-1] >= y0 >= y[i0]. """
    i = (i_beg + 1) % y.size
    while True:
        if y[i - 1] >= y0 >= y[i]:
            i0 = i
            break
        else:
            i += 1
            if i >= y.size:
                i -= y.size
    return i0


def detect_left_peak(y, i0, relative_height):
    n = y.size
    ymax = y.max()
    peak = []
    vally = []
    i = i0
    while True:
        if y[i - 1] >= y[i] and y[i - 1] > y[i - 2]:
            peak.append(i - 1)
            if len(peak) == 2 and len(vally) >= 1:
                if y[peak[0]] > y[peak[1]]:
                    del peak[1]
                else:
                    del peak[0]
                del vally[-1]
        elif y[i - 1] < y[i] and y[i - 1] <= y[i - 2]:
            vally.append(i - 1)
            if len(peak) == 1:
                dy = y[peak[0]] - y[vally[-1]]
                dx = cal_dis(peak[0], vally[-1], n)
                # if (dx >= 4 or dy >= relative_height) and \
                #         y[peak[0]] > 0.33 * ymax:
                #     break
                y_peak = y[peak[0]]
                if y_peak > 3 and (dy > 0.2 * relative_height or dx >= 3):
                    break
                elif y_peak > 2 and (dy > 0.4 * relative_height or dx >= 4):
                    break
                elif y_peak > 0.33 * ymax and (dy >= relative_height or
                                               dx >= 4):
                    break

            elif len(peak) > 1:
                print("Error when detect left peak, peak = %d" % len(peak))
                plt.plot(y)
                plt.axvline(peak[0], c="r")
                plt.axvline(peak[1], c="b")
                plt.axvline(i0, c="k", linestyle="--")
                plt.show()
                plt.close()
                sys.exit()
        i -= 1
        i = i % n
    return peak[0]


def detect_right_peak(y, i0, vally_thresh, relative_height, max_step=40):
    n = y.size
    ymax = y.max()
    peak = []
    vally = []
    i = i0
    iter_count = 0
    while True:
        if y[i - 1] >= y[i] and y[i - 1] > y[i - 2]:
            peak.append(i - 1)
            if len(peak) == 1 and len(vally) == 0:
                j = i - 2
                while True:
                    if y[j] < y[(j + 1) % n] and y[j] <= [j - 1]:
                        vally.append(j)
                        break
                    else:
                        j -= 1
                        j = j % n
            elif len(peak) == 2 and len(vally) == 2:
                if y[vally[0]] > y[vally[1]]:
                    del peak[0]
                    del vally[0]
                else:
                    if y[peak[0]] > y[peak[1]]:
                        del peak[1]
                    else:
                        del peak[0]
                    del vally[1]
        elif y[i - 1] < y[i] and y[i - 1] <= y[i - 2]:
            vally.append(i - 1)
            if len(peak) == 1:
                dx_left = cal_dis(peak[0], vally[0], n)
                dx_right = cal_dis(vally[1], peak[0], n)
                dy_left = y[peak[0]] - y[vally[0]]
                if dx_right > 4 and y[vally[0]] > vally_thresh:
                    y_peak = y[peak[0]]
                    if y_peak > 3 and (dy_left > 0.2 * relative_height or
                                       dx_left >= 3):
                        break
                    elif y_peak > 2 and (dy_left > 0.4 * relative_height or
                                         dx_left >= 4):
                        break
                    elif y_peak > 0.33 * ymax and (dy_left > relative_height or
                                                   dx_left >= 4):
                        break
            elif len(peak) > 1:
                print("Error when dectect right peak, peak = %d" % (len(peak)))
                plt.plot(y)
                plt.axvline(peak[0], c="r")
                plt.axvline(peak[1], c="b")
                for v in vally:
                    plt.axvline(v, c="g", linestyle="-.")
                plt.axvline(i0, c="k", linestyle="--")
                plt.show()
                plt.close()
                sys.exit()
        i += 1
        i = i % n
        iter_count += 1
        if iter_count >= max_step:
            return None
    return peak[0]


def find_first_row(rho, debug=0):
    for row, rho_x in enumerate(rho):
        rho_h = 0.5 * rho_x.max()
        count = 0
        for i in range(rho_x.size):
            if rho_x[i - 1] >= rho_h >= rho_x[i]:
                count += 1
                idx_h = i
        if count == 1:
            idx_peak = detect_right_peak(
                rho_x,
                idx_h,
                vally_thresh=0.2,
                relative_height=0.1,
                max_step=50)
            if idx_peak is None:
                idx_peak = detect_left_peak(rho_x, idx_h, 0.5)
                if rho_x[idx_peak] >= 4:
                    return row, idx_peak


def get_next_idx_h(rho_x, idx_h_pre, rho_h_pre=None, debug=False):
    idx_peak = detect_right_peak(rho_x, idx_h_pre, rho_h_pre * 0.5, 0.5)
    if idx_peak is None:
        idx_peak = detect_left_peak(rho_x, idx_h_pre, 0.5)
    rho_h = rho_x[idx_peak] * 0.5
    idx_h = get_idx_nearest(rho_x, idx_peak, rho_h)
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
        if debug > 1 and 60 < row < 100:
            flag = True
        else:
            flag = False
        row_pre = (row - drow) % nrows
        idx_h[row], rho_h[row] = get_next_idx_h(
            rho[row], idx_h[row_pre], rho_h[row_pre], debug=flag)
        if debug:
            print("row = ", row)
    return idx_h, rho_h


def find_interface(rho, sigma, debug=0, leftward=False):
    rho_s = gaussian_filter(rho, sigma=sigma, mode="wrap")
    # start_row, start_idx_h = find_ini_row(rho_s, debug=debug)
    start_row, idx_peak = find_first_row(rho_s)
    if idx_peak is not None:
        start_idx_h = get_idx_nearest(rho_s[start_row], idx_peak,
                                      rho_s[start_row, idx_peak] * 0.5)
    if debug:
        plt.plot(rho_s[start_row])
        plt.axvline(start_idx_h)
        plt.axvline(idx_peak)
        plt.axhline(rho_s[start_row, idx_peak] * 0.5)
        plt.show()
        plt.close()
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
    debug = 1
    t_beg = 2164
    t_end = 2165
    w1 = []
    w2 = []
    for i, frame in enumerate(snap.gene_frames(t_beg, t_end)):
        x, y, theta = frame
        rho = load_snap.coarse_grain2(x, y, theta, Lx=Lx, Ly=Ly).astype(float)
        xh, rho_h = find_interface(rho, sigma=[5, 1], debug=debug)
        # xh1, rho_h1 = find_interface(rho, sigma=[5, 1], leftward=True)
        xh2, rho_h2 = half_rho.find_interface(rho, sigma=[5, 1])
        xh = half_rho.untangle(xh, Lx)
        # xh1 = half_rho.untangle(xh1, Lx)
        xh2 = half_rho.untangle(xh2, Lx)
        yh = np.linspace(0, Ly - 1, Ly)

        print(i)
        if debug > 0:
            rho[rho > 4] = 4
            plt.imshow(rho.T, interpolation="none", origin="lower")
            plt.plot(yh, xh, "k")
            # plt.plot(yh, xh1, "w--")
            plt.plot(yh, xh2, "r")
            # plt.plot(yh, xh3, "g")
            plt.show()
            plt.close()
        w1.append(np.var(xh))
        w2.append(np.var(xh2))
    plt.plot(w1, label="1")
    plt.plot(w2, label="2")
    plt.legend()
    plt.show()
