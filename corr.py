""" Cal two-point correlation function:

    C(r, t) = <[\sigma h(r0, t0) - \sigma(r0+r, t0+t)]^2>,
    with \sigma h = h - <h>.

    G(r) = C(r, 0) \sim r^{2 \alpha}
    C_s(t) = C(0, t) \sim t^{2 \beta}
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


def cal_G(h, r):
    """ Cal G(r) = C(r, 0) to determine the roughening exponent alpha.

        Parameters:
        --------
        h: np.ndarray
            Height of the interface.
        r: np.ndarray
            Distance, must be int.

        Returns:
        --------
        G: np.ndarray
            The same shape as r.
    """
    G = np.zeros(r.size)
    for i, dr in enumerate(r):
        dh = h - np.roll(h, dr)
        G[i] = np.var(dh)
    return G


if __name__ == "__main__":
    import load_snap
    from half_rho import untangle
    import snake
    import half_peak
    os.chdir(r"D:\tmp")
    Lx = 180
    Ly = 1000
    sigma = [1, 1]
    r = np.round(np.logspace(2, 19, 18, base=np.sqrt(2))).astype(int)
    G1 = np.zeros(r.size)
    G2 = np.zeros(r.size)
    count = 0
    snap = load_snap.RawSnap(r"so_%g_%g_%d_%d_%d_%d_%d.bin" %
                             (0.35, 0, Lx, Ly, Lx * Ly, 2000, 1234))
    t_beg = 250
    t_end = None
    for i, frame in enumerate(snap.gene_frames(t_beg, t_end)):
        x, y, theta = frame
        rho = load_snap.coarse_grain2(
            x, y, theta, Lx=Lx, Ly=Ly, ncols=Lx, nrows=Ly).astype(float)
        xh1, rho_h = half_peak.find_interface(rho, sigma=sigma)
        rho_s = gaussian_filter(rho, sigma=sigma)
        xh2, yh2 = snake.find_interface(
            rho_s, 0.5, 0.1, 0.25, 400, rho_h, dx=5)
        xh1 = untangle(xh1, Lx)
        xh2 = untangle(xh2, Lx)
        G1 += cal_G(xh1, r)
        G2 += cal_G(xh2, r)
        count += 1
        print("i = ", i)
    G1 /= count
    G2 /= count
    plt.plot(r, G1)
    plt.plot(r, G2)
    plt.xscale("log")
    plt.yscale("log")
    plt.show()
    plt.close()
    for i in range(G1.size):
        print(r[i], G1[i], G2[i])

