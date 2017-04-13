import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import os


def read(file):
    with open(file) as f:
        lines = f.readlines()
    n = len(lines)
    phi = np.zeros(n)
    for i in range(n):
        s = lines[i].split("\t")
        phi[i] = float(s[1])
    return phi


def get_phi_mean(Lxs, Lys, ncut=2500, eta=0.35, eps=0, rho=1, seed=1234):
    phi_mean = np.zeros((len(Lxs), len(Lys)))
    for i, Lx in enumerate(Lxs):
        for j, Ly in enumerate(Lys):
            file = "p_%g_%g_%g_%d_%d_%d.dat" % (eta, eps, rho, Lx, Ly, seed)
            phi = read(file)
            phi_mean[i, j] = np.mean(phi[ncut:])
    return phi_mean


def phi_vs_Lx_and_Ly():
    Lxs = [150, 160, 180, 200, 220]
    Lys = [
        150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 700, 800, 900, 1000
    ]

    phi = get_phi_mean(Lxs, Lys)
    plt.subplot(121)
    for i, Lx in enumerate(Lxs):
        plt.plot(Lys, phi[i], "-o", label=r"$L_x=%d$" % Lx)
    plt.xscale("log")
    plt.yscale("log")
    plt.legend(loc="best")

    plt.subplot(122)
    for j, Ly in enumerate(Lys):
        plt.plot(Lxs, phi[:, j], "-o", label=r"$L_y=%d$" % Ly)
    plt.legend(loc="best")
    plt.show()
    plt.close()


def moving_average(phi: np.ndarray, window: int = 500):
    t = []
    phi_mean = []
    for i in range(phi.size - window):
        t.append((window // 2 + i) * 100)
        phi_mean.append(phi[i: i + window].mean())
    plt.subplot(211)
    plt.plot(t, phi_mean)
    plt.subplot(212)
    phi_gauss = gaussian_filter1d(phi, sigma=200)
    plt.plot(phi_gauss)
    plt.show()
    plt.close()


if __name__ == "__main__":
    os.chdir("data/phi")
    file = "p_0.35_0_1_220_1000_1234.dat"
    phi = read(file)
    plt.plot(phi)
    plt.show()
    plt.close()
    moving_average(phi)
    # phi_vs_Lx_and_Ly()

