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


def find_idx_max(rho_x, pre_idx_max, idx_range=20, is_first_row=False):
    def get_max(i_beg, i_end):
        tmp = np.array([
            rho_x[i - n] if i >= n else rho_x[i] for i in range(i_beg, i_end)
        ])
        return np.argmax(tmp) + i_beg

    import matplotlib.pyplot as plt
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
        if is_first_row:
            rho_h = rho_x[idx] * 0.5
            xh_count = 0
            for i in range(n):
                if rho_x[i - 1] >= rho_h >= rho_x[i]:
                    xh_count += 1
            if xh_count > 1:
                idx = None
        else:
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


def find_half_rho(rho_x, idx_max, x_pre=None, debug=False):
    def show():
        plt.plot(rho_x, "-o")
        plt.axhline(rho_max, color="r")
        plt.axhline(rho_h, color="g")
        plt.axvline(idx_max, color="r", linestyle="--")
        plt.axvline(xh, color="g", linestyle=":")
        plt.show()
        plt.close()

    import matplotlib.pyplot as plt
    rho_max = rho_x[idx_max]
    rho_h = 0.5 * rho_max
    n = rho_x.size
    i = idx_max + 1
    iter_count = 0
    while True:
        rho0 = rho_x[(i - 1) % n]
        rho1 = rho_x[i % n]
        if rho0 >= rho_h >= rho1:
            x0 = i - 0.5
            x1 = i + 0.5
            xh = x0 - (rho0 - rho_h) / (rho0 - rho1) * (x0 - x1)
            if x_pre is not None:
                dx = abs(x_pre - xh)
                dx = min(dx, n - dx)
                if dx < 5:
                    if debug:
                        show()
                    if xh >= rho_x.size:
                        xh -= rho_x.size
                    return xh, rho_h
                else:
                    i += 1
                    iter_count += 1
            else:
                if debug:
                    show()
                if xh >= rho_x.size:
                    xh -= rho_x.size
                return xh, rho_h
        else:
            i += 1
            iter_count += 1
        if iter_count >= n:
            if platform.system() is "Windows":
                plt.plot(rho_x, "-o")
                plt.axhline(rho_max, color="r")
                plt.axhline(rho_h, color="g")
                plt.axvline(idx_max, color="r", linestyle="--")
                plt.axvline(xh, color="g", linestyle=":")
                plt.axvline(x_pre, color="b", linestyle="-.")
                plt.title("iteration count = %d, $d_x=%g$" % (iter_count, dx))
                plt.show()
                plt.close()
            print("Too many iterations when find half rho.")
            sys.exit()


def find_interface(rho, sigma, debug=False):
    import matplotlib.pyplot as plt
    rho_s = gaussian_filter(rho, sigma=sigma, mode="wrap")
    nrows = rho_s.shape[0]
    xh = np.zeros(nrows)
    rho_h = np.zeros(nrows)

    # find starting index
    start_row = 0
    idx_max_pre = np.argmax(np.mean(rho_s, axis=0))
    for row, rhox in enumerate(rho_s):
        idx_max = find_idx_max(rhox, idx_max_pre, is_first_row=True)
        if idx_max is not None:
            plt.plot(rhox, "-o")
            plt.show()
            plt.close()
            start_row = row
            xh[row], rho_h[row] = find_half_rho(rhox, idx_max, debug=debug)
            print("starting row: ", row)
            break

    # find xh, rho_h for each row
    for row in range(start_row + 1, start_row + nrows):
        if row >= nrows:
            row -= nrows
        rhox = rho_s[row]
        idx_max = find_idx_max(rhox, idx_max_pre)

        # if row >= 1:
        #     sub_debug = True
        # else:
        #     sub_debug = False
        print(row)
        xh[row], rho_h[row] = find_half_rho(
            rhox, idx_max, x_pre=xh[row - 1], debug=False)
        idx_max_pre = idx_max

    if debug:
        plt.imshow(rho_s.T, origin="lower", interpolation="none")
        yh = np.linspace(0.5, nrows - 0.5, nrows)
        plt.plot(yh, xh, "k")
        plt.show()
        plt.close()
    return xh, rho_h


if __name__ == "__main__":
    pass
