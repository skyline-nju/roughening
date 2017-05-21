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


def get_log_time(t_end, exponent, show=False):
    ts = [1]
    t_cur = 1
    while True:
        t_cur = t_cur * exponent
        if t_cur > t_end:
            break
        t_cur_int = int(np.round(t_cur))
        if t_cur_int > ts[-1]:
            ts.append(t_cur_int)
    if show:
        plt.subplot(121)
        plt.plot(ts, "-o")
        plt.subplot(122)
        plt.plot(ts, "-o")
        plt.yscale("log")
        plt.suptitle("n = %d" % (len(ts)))
        plt.show()
        plt.close()
    return ts


def get_phi_mean(Lxs, Lys, ncut=5000, eta=0.35, eps=0, rho=1, seed=1234):
    phi_mean = np.zeros((len(Lxs), len(Lys)))
    for i, Lx in enumerate(Lxs):
        for j, Ly in enumerate(Lys):
            file = "p_%g_%g_%g_%d_%d_%d.dat" % (eta, eps, rho, Lx, Ly, seed)
            phi = read(file)
            phi_mean[i, j] = np.mean(phi[ncut:])
    return phi_mean


def phi_vs_Lx_and_Ly():
    Lxs = [150, 160, 180, 200, 220]
    # Lys = [
    #     150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 700, 800, 900, 1000
    # ]
    Lys = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

    phi = get_phi_mean(Lxs, Lys)
    plt.figure(figsize=(12, 4.2))
    plt.subplot(131)
    for i, Lx in enumerate(Lxs):
        plt.plot(Lys, phi[i], "-o", label=r"$L_x=%d$" % Lx)
    plt.legend(loc="best", labelspacing=0)
    plt.xlabel(r"$L_y$")
    plt.ylabel(r"$\langle \phi \rangle_t$")
    plt.title("(a)")
    # plt.xscale("log")
    # plt.yscale("log")

    plt.subplot(132)
    for i, Lx in enumerate(Lxs):
        plt.loglog(Lys, phi[i], "-o", label=r"$L_x=%d$" % Lx)
    plt.xlabel(r"$L_y$")
    plt.title("(b)")
    plt.legend(loc="best", labelspacing=0)

    plt.subplot(133)
    for j, Ly in enumerate(Lys):
        plt.plot(Lxs, phi[:, j], "-s", label=r"$L_y=%d$" % Ly)
    plt.tight_layout()
    plt.xlabel(r"$L_x$")
    plt.legend(
        loc="best",
        labelspacing=0,
        ncol=2,
        columnspacing=0.5,
        fontsize="small")
    plt.title("(c)")
    plt.show()
    plt.close()


def moving_average(phi: np.ndarray, window: int=100):
    t = []
    phi_mean = []
    for i in range(phi.size - window):
        t.append((window // 2 + i) * 100)
        phi_mean.append(phi[i:i + window].mean())
    plt.subplot(211)
    plt.plot(t, phi_mean)
    plt.subplot(212)
    phi_gauss = gaussian_filter1d(phi, sigma=20)
    plt.plot(phi_gauss)
    plt.show()
    plt.close()


if __name__ == "__main__":
    os.chdir("data/phi")
    phi_vs_Lx_and_Ly()