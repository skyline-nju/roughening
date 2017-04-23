""" Cal two-point correlation function:

    C(r, t) = <[\sigma h(r0, t0) - \sigma h(r0+r, t0+t)]^2>,
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
    if h.ndim == 1:
        for i, dr in enumerate(r):
            dh = h - np.roll(h, dr)
            G[i] = np.mean(dh**2)
    else:
        for i, dr in enumerate(r):
            dh = h - np.roll(h, dr, axis=1)
            G[i] = np.mean(dh**2)
    return G


def cal_Cs(h):
    nrows, ncols = h.shape
    Cs = []
    t = []
    sigma_h = np.array([hi - np.mean(hi) for hi in h])
    print(sigma_h.shape)
    for drow in range(1, int(nrows / 2)):
        sum_Cs = 0
        count = 0
        for j in range(nrows - drow):
            sum_Cs += np.mean((sigma_h[j + drow] - sigma_h[j])**2)
            count += 1
        Cs.append(sum_Cs / count)
        t.append(drow)
    return np.array(t), np.array(Cs)


def handle_raw_snap():
    import load_snap
    from half_rho import untangle
    import snake
    import half_peak
    os.chdir(r"D:\tmp")
    Lx = 180
    Ly = 1000
    sigma = [5, 1]
    r = np.round(np.logspace(2, 18, 17, base=np.sqrt(2))).astype(int)
    G1 = np.zeros(r.size)
    G2 = np.zeros(r.size)
    count = 0
    snap = load_snap.RawSnap(r"so_%g_%g_%d_%d_%d_%d_%d.bin" %
                             (0.35, 0, Lx, Ly, Lx * Ly, 2000, 1234))
    n = snap.get_num_frames()
    print("n=", n)
    t_beg = 250
    t_end = 300
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
    print(count)


if __name__ == "__main__":
    os.chdir("data\interface")
    r = np.round(np.logspace(2, 18, 17, base=np.sqrt(2))).astype(int)
    Lx = 200
    Ly = 1000
    eps = 0
    sigma_y = 1
    file = "so_0.35_%g_%d_%d_%d_2000_1234_%d.npz" % (eps, Lx, Ly, Lx * Ly,
                                                     sigma_y)
    data = np.load(file)
    h1 = data["h1"][250:]
    h2 = data["h2"][250:]
    r = np.round(np.logspace(2, 20, 19, base=np.sqrt(2))).astype(int)
    G1 = cal_G(h1, r)
    G2 = cal_G(h2, r)
    plt.plot(r, G1)
    plt.plot(r, G2)
    plt.xscale("log")
    plt.yscale("log")
    plt.show()
    plt.close()
    for i in range(r.size):
        print(r[i], G1[i], G2[i])
