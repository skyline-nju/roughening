""" Show each method to detect the interface of band. """

import os
import numpy as np
import matplotlib.pyplot as plt
import load_snap
import half_peak
import half_rho
# import cluster

if __name__ == "__main__":
    os.chdir(r"D:\tmp")
    Lx = 180
    Ly = 1000
    file = r"so_0.35_0_%d_%d_%d_2000_1234.bin" % (Lx, Ly, Lx * Ly)
    snap = load_snap.RawSnap(file)
    for frame in snap.gene_frames(1758, 1759):
        x, y, theta = frame
        # cluster.show_cluster(x, y, Lx, Ly)
        plt.figure(figsize=(10, 4))
        rho = load_snap.coarse_grain2(x, y, theta, Lx=Lx, Ly=Ly).astype(float)
        xh, rho_h = half_peak.find_interface(rho, sigma=[1, 1])
        xh2, rho_h2 = half_peak.find_interface(rho, sigma=[5, 1])
        xh3, rho_h3 = half_peak.find_interface(rho, sigma=[10, 1])
        xh = half_rho.untangle(xh, Lx)
        xh2 = half_rho.untangle(xh2, Lx)
        xh3 = half_rho.untangle(xh3, Lx)
        yh = np.linspace(0, Ly - 1, Ly)

        rho = load_snap.coarse_grain(x, y, Lx=Lx, Ly=Ly)
        rho[rho > 5] = 5
        plt.imshow(rho.T, origin="lower", cmap="viridis", aspect="auto")
        plt.plot(yh, xh, "k", label=r"$\sigma_y=1, w^2=%.4f$" % (np.var(xh)))
        plt.plot(yh, xh2, "r", label=r"$\sigma_y=5, w^2=%.4f$" % (np.var(xh2)))
        plt.plot(yh, xh3, "w", label=r"$\sigma_y=10, w^2=%.4f$" % (np.var(xh3)))
        plt.xlabel(r"$y$")
        plt.ylabel(r"$x$")
        plt.title(r"$\eta=0.18, \epsilon=0, \rho_0=1, L_x=%d, L_y=%d$" %
                  (Lx, Ly))
        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.close()
