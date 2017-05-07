""" Calculate averaged profile of rho_y(x).

    For rho_y(x), find x=h where rho_y(x) = max(rho_y)/2, and shift rho_y(x)
    so that rho_y = max(rho_y)/2 at x=xc. Then average rho_y over y and time.
"""
import os
import sys
import numpy as np
import matplotlib
import load_snap
from half_peak import find_interface
import platform
if platform.system() is "Windows":
    import matplotlib.pyplot as plt
    os.chdir(r"D:\tmp")
else:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.chdir(r"snap_one")


def ave_one_frame(rho, xh, xc=100):
    nrows, ncols = rho.shape
    mean_rhox = np.zeros(ncols)
    x0 = np.arange(ncols) + 0.5
    for row in range(nrows):
        x_new = x0 - xc + xh[row]
        mean_rhox += np.interp(x_new, x0, rho[row], period=ncols)
    mean_rhox /= nrows
    return mean_rhox


def time_ave(Lx, Ly, sigma_y=100, show=False, out=False, eps=0, dt=1):
    file = r"so_0.35_%g_%d_%d_%d_2000_1234.bin" % (eps, Lx, Ly, Lx * Ly)
    snap = load_snap.RawSnap(file)
    if not isinstance(sigma_y, list):
        nrows = 1
        sigma_y = [sigma_y]
    else:
        nrows = len(sigma_y)
    ncols = Lx
    rho_mean = np.zeros((nrows, ncols))
    count = np.zeros(nrows, int)
    x0 = np.arange(Lx) + 0.5
    for frame in snap.gene_frames(beg_idx=300, interval=dt):
        x, y, theta = frame
        rho = load_snap.coarse_grain2(x, y, theta, Lx=Lx, Ly=Ly).astype(float)
        rho_real = load_snap.coarse_grain(x, y, Lx=Lx, Ly=Ly)
        for i, sy in enumerate(sigma_y):
            try:
                xh, rho_h = find_interface(rho, sigma=[sy, 1])
                rho_mean[i] += ave_one_frame(rho_real, xh)
                count[i] += 1
            except:
                pass
        print("t=%d" % (dt * count[-1]))
    for i in range(nrows):
        rho_mean[i] /= count[i]
    if show:
        for i in range(nrows):
            plt.plot(x0, rho_mean[i])
        plt.show()
        plt.close()
    if out:
        outfile = "avePeak_%g_%d_%d.dat" % (eps, Lx, Ly)
        with open(outfile, "w") as f:
            for i in range(ncols):
                line = "%f" % (x0[i])
                for j in range(nrows):
                    line += "\t%.8f" % (rho_mean[j, i])
                line += "\n"
                f.write(line)


if __name__ == "__main__":
    if platform.system() is "Windows":
        time_ave(150, 400, sigma_y=[10, 20], out=True, dt=100)
    else:
        sigma_y = [5, 10, 15, 20]
        if len(sys.argv) == 3:
            Lx = int(sys.argv[1])
            Ly = int(sys.argv[2])
            eps = 0
        elif len(sys.argv) == 4:
            Lx = int(sys.argv[1])
            Ly = int(sys.argv[2])
            eps = float(sys.argv[3])
        time_ave(Lx, Ly, sigma_y, out=True, eps=eps)
