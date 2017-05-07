import os
import sys
import numpy as np
import matplotlib
import load_snap
from half_peak import find_interface
from half_rho import untangle
import platform
if platform.system() is "Windows":
    import matplotlib.pyplot as plt
    os.chdir(r"D:\tmp")
else:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.chdir(r"snap_one")


def cal_spectrum(Lx, Ly, sigma_y=10, show=False, out=False, eps=0, dt=1):
    file = r"so_0.35_%g_%d_%d_%d_2000_1234.bin" % (eps, Lx, Ly, Lx * Ly)
    snap = load_snap.RawSnap(file)
    if not isinstance(sigma_y, list):
        nrows = 1
        sigma_y = [sigma_y]
    else:
        nrows = len(sigma_y)
    ncols = Ly//2 + 1
    q = np.arange(ncols) / Ly
    spectrum = np.zeros((nrows, ncols))
    count = np.zeros(nrows, int)
    for frame in snap.gene_frames(beg_idx=300, interval=dt):
        x, y, theta = frame
        rho = load_snap.coarse_grain2(x, y, theta, Lx=Lx, Ly=Ly).astype(float)
        for i, sy in enumerate(sigma_y):
            try:
                xh, rho_h = find_interface(rho, sigma=[sy, 1])
                xh = untangle(xh, Lx)
                h = xh - np.mean(xh)
                hq = np.fft.rfft(h) / Ly
                A2 = np.abs(hq) ** 2
                spectrum[i] += A2
                count[i] += 1
            except:
                pass
        print("t=%d" % (dt * count[-1]))
    for i in range(nrows):
        spectrum[i] /= count[i]
    if show:
        plt.loglog(q, spectrum)
        plt.show()
        plt.close()
    if out:
        if isinstance(sigma_y, list):
            outfile = "hq_%g_%d_%d.dat" % (eps, Lx, Ly)
        else:
            outfile = "hq_%g_%d_%d_%d.dat" % (eps, Lx, Ly, sigma_y)
        with open(outfile, "w") as f:
            for i in range(q.size):
                line = "%f" % (q[i] ** 2)
                for j in range(nrows):
                    line += "\t%.8f" % (spectrum[j, i])
                line += "\n"
                f.write(line)


if __name__ == "__main__":
    if platform.system() is "Windows":
        cal_spectrum(150, 400, sigma_y=15, out=True, dt=100)
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
        cal_spectrum(Lx, Ly, sigma_y, out=True, eps=eps)
