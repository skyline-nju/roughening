import numpy as np
import matplotlib.pyplot as plt
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


def get_phi_mean(Lxs, Lys, ncut=2000, eta=0.35, eps=0, rho=1, seed=1234):
    phi_mean = np.zeros((len(Lxs), len(Lys)))
    for i, Lx in enumerate(Lxs):
        for j, Ly in enumerate(Lys):
            file = "p_%g_%g_%g_%d_%d_%d.dat" % (eta, eps, rho, Lx, Ly, seed)
            phi = read(file)
            phi_mean[i, j] = np.mean(phi[ncut:])
    return phi_mean


if __name__ == "__main__":
    os.chdir("data/phi")

    Lxs = [150, 160, 180, 200, 220]
    Lys = [200, 300, 400, 500, 600, 700, 800, 900, 1000]

    phi = get_phi_mean(Lxs, Lys)
    plt.subplot(121)
    for i, Lx in enumerate(Lxs):
        plt.plot(Lys, phi[i], "-o", label=r"$L_x=%d$" % Lx)
    # plt.xscale("log")
    # plt.yscale("log")
    plt.legend(loc="best")

    plt.subplot(122)
    for j, Ly in enumerate(Lys):
        plt.plot(Lxs, phi[:, j], "-o", label=r"$L_y=%d$" % Ly)
    plt.legend(loc="best")
    plt.show()
    plt.close()
