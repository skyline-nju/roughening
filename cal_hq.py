import os
import numpy as np
import matplotlib.pyplot as plt
import load_snap
from half_peak import find_interface
from half_rho import untangle


def cal_spectrum(Lx, Ly, sigma_y=10, show=False, out=False):
    os.chdir(r"D:\tmp")
    file = r"so_0.35_0_%d_%d_%d_2000_1234.bin" % (Lx, Ly, Lx * Ly)
    snap = load_snap.RawSnap(file)
    spectrum = np.zeros(Ly//2 + 1)
    count = 0
    dt = 1
    for frame in snap.gene_frames(beg_idx=300, interval=dt):
        x, y, theta = frame
        rho = load_snap.coarse_grain2(x, y, theta, Lx=Lx, Ly=Ly).astype(float)
        # yh = np.linspace(0.5, Ly-0.5, Ly)
        xh, rho_h = find_interface(rho, sigma=[sigma_y, 1])
        xh = untangle(xh, Lx)
        h = xh - np.mean(xh)
        hq = np.fft.rfft(h)
        A2 = np.abs(hq) ** 2
        spectrum += A2
        print("t=%d" % (dt * count))
        count += 1
    spectrum /= count
    q = np.arange(spectrum.size)
    if show:
        plt.loglog(q, spectrum)
        plt.show()
        plt.close()
    if out:
        with open("hq_%d_%d_%d.dat" % (Lx, Ly, sigma_y), "w") as f:
            for i in range(q.size):
                f.write("%d\t%f\n" % (q[i] ** 2, spectrum[i]))


if __name__ == "__main__":
    cal_spectrum(150, 400, 10, out=True)
