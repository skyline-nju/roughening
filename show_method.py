""" Show each method to detect the interface of band. """

import os
import numpy as np
import matplotlib.pyplot as plt
import load_snap
import half_peak
import half_rho
import snake
from scipy.ndimage import gaussian_filter
from cal_width_old import isoline
import cluster

if __name__ == "__main__":
    os.chdir(r"D:\tmp")
    Lx = 180
    Ly = 1000
    file = r"so_0.35_0_%d_%d_%d_2000_1234.bin" % (Lx, Ly, Lx * Ly)
    snap = load_snap.RawSnap(file)
    for frame in snap.gene_frames(1758, 1759):
        x, y, theta = frame
        cluster.show_cluster(x, y, Lx, Ly)
        plt.figure(figsize=(14, 3.5))
        rho = load_snap.coarse_grain2(x, y, theta, Lx=Lx, Ly=Ly).astype(float)
        xh, rho_h = half_peak.find_interface(rho, sigma=[1, 1])
        xh2, rho_h2 = half_peak.find_interface(rho, sigma=[5, 1])
        xh3, rho_h3 = half_peak.find_interface(rho, sigma=[10, 1])
        xh = half_rho.untangle(xh, Lx)
        xh2 = half_rho.untangle(xh2, Lx)
        xh3 = half_rho.untangle(xh3, Lx)
        yh = np.linspace(0, Ly - 1, Ly)

        rho2 = load_snap.coarse_grain(x, y, Lx=Lx, Ly=Ly)
        rho2[rho2 > 5] = 5
        plt.imshow(rho2.T, origin="lower", cmap="viridis", aspect="auto")
        template = r"$sigma_y%d, w^2=%.4f$"
        plt.plot(yh, xh, "r", label=template % (1, np.var(xh)), lw=2)
        plt.plot(yh, xh2, "w", label=template % (5, np.var(xh2)), lw=2)
        plt.plot(yh, xh3, "k", label=template % (10, np.var(xh3)), lw=2)
        plt.xlabel(r"$y$")
        plt.ylabel(r"$x$")
        plt.title(r"$\eta=0.18, \epsilon=0, \rho_0=1, L_x=%d, L_y=%d$" %
                  (Lx, Ly))
        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.close()

        plt.figure(figsize=(14, 3.5))
        plt.imshow(rho2.T, origin="lower", cmap="viridis", aspect="auto")
        var, mean, xm, yc, h = isoline(rho, Lx, Ly, Lx, Ly // 10)
        plt.plot(yc, h, "rs-", label=r"$l_y=10, w^2=%f$" % var, ms=4)
        var, mean, xm, yc, h = isoline(rho, Lx, Ly, Lx, Ly // 20)
        plt.plot(yc, h, "wo-", label=r"$l_y=20, w^2=%f$" % var, ms=4)
        plt.xlabel(r"$y$")
        plt.ylabel(r"$x$")
        plt.title(r"$\eta=0.18, \epsilon=0, \rho_0=1, L_x=%d, L_y=%d$" %
                  (Lx, Ly))
        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.close()

        plt.figure(figsize=(14, 3.5))
        rho_s = gaussian_filter(rho, sigma=[1, 1])
        xl, yl = snake.find_interface(rho_s, 1, 0.5, 0.25, 500, rho_h, dx=5)
        xl2, yl2 = snake.find_interface(rho_s, 2, 0.5, 0.25, 500, rho_h, dx=5)
        xl3, yl3 = snake.find_interface(rho_s, 4, 0.5, 0.25, 500, rho_h, dx=5)

        plt.imshow(rho2.T, origin="lower", cmap="viridis", aspect="auto")
        template = r"$\alpha=%g, \beta=%g, w^2=%f$"
        plt.plot(yl, xl, "r", label=template % (1, 0.5, np.var(xl)), lw=2)
        plt.plot(yl2, xl2, "w", label=template % (2, 0.5, np.var(xl2)), lw=2)
        plt.plot(yl3, xl3, "k", label=template % (4, 0.5, np.var(xl3)), lw=2)

        plt.xlabel(r"$y$")
        plt.ylabel(r"$x$")
        plt.title(r"$\eta=0.18, \epsilon=0, \rho_0=1, L_x=%d, L_y=%d$" %
                  (Lx, Ly))
        plt.legend()
        plt.tight_layout()
        plt.show()
