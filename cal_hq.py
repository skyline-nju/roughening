""" Decompose the profile h into Fourier modes hq.

    h(y) = /sum_q{h_q /exp{iqy}} with
    h_q = 1/L_y /quad_0^L_y{dy h(y) /exp{-iqy}}
"""
import os
import sys
import numpy as np
import matplotlib
import load_snap
from half_peak import find_interface
from half_rho import untangle
from w_Ly_short_time import add_line
import platform
if platform.system() is "Windows":
    import matplotlib.pyplot as plt
else:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt


def cal_spectrum(Lx, Ly, sigma_y=10, show=False, out=False, eps=0, dt=1):
    file = r"so_0.35_%g_%d_%d_%d_2000_1234.bin" % (eps, Lx, Ly, Lx * Ly)
    snap = load_snap.RawSnap(file)
    if not isinstance(sigma_y, list):
        nrows = 1
        sigma_y = [sigma_y]
    else:
        nrows = len(sigma_y)
    ncols = Ly // 2 + 1
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
                A2 = np.abs(hq)**2
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
        outfile = "hq_%g_%d_%d.dat" % (eps, Lx, Ly)
        with open(outfile, "w") as f:
            for i in range(q.size):
                line = "%f" % (q[i]**2)
                for j in range(nrows):
                    line += "\t%.8f" % (spectrum[j, i])
                line += "\n"
                f.write(line)


def handle_snap():
    if platform.system() is "Windows":
        os.chdir(r"D:\tmp")
        cal_spectrum(150, 400, sigma_y=15, out=True, dt=100)
    else:
        os.chdir(r"snap_one")
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


def read(Lx, Ly):
    file = r"data\hq\hq_%d_%d.dat" % (Lx, Ly)
    with open(file) as f:
        lines = f.readlines()
        nrows = len(lines)
        ncols = 5
        data = np.zeros((nrows, ncols))
        for row, line in enumerate(lines):
            for col, s in enumerate(line.replace("\n", "").split("\t")):
                data[row, col] = float(s)
        data = data.T
        q2 = data[0]
        sigma_y = [5, 10, 15, 20]
        hq2 = {sigma_y[i]: data[i + 1] for i in range(len(sigma_y))}
        return q2, hq2


def plot_varied_sigma_y(Lx, Ly, eta=0.18, eps=0, ax=None, save=False):
    if ax is None:
        flag_show = True
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
    else:
        flag_show = False
    q2, hq2 = read(Lx, Ly)
    for sigma_y in hq2:
        ax.plot(q2, hq2[sigma_y], "o", label=r"$\sigma_y=%d$" % sigma_y, ms=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$(q / 2\pi)^2$", fontsize="large")
    ax.set_ylabel(r"$\langle |h_q|^2\rangle_t$", fontsize="large")
    llabel = r"${\rm slope}=-1$"
    add_line(ax, 0, 0.75, 1, -1, scale="log", label=llabel, xl=0.3, yl=0.4)
    add_line(ax, 0, 0.92, 1, -1, scale="log")
    add_line(ax, 0.1, 1, 1, -1, scale="log")
    ax.legend()
    ax.set_title(r"$\eta=0.18, \epsilon=0, \rho_0=1, L_x=%d, L_y=%d$" %
                 (Lx, Ly))
    if flag_show:
        plt.tight_layout()
        if not save:
            plt.show()
        else:
            plt.savefig(r"data\hq\varied_sy_%d_%d.pdf" % (Lx, Ly))
        plt.close()


def plot_varied_Ly(Lx,
                   sigma_y,
                   Lys,
                   eta=0.18,
                   eps=0,
                   ax=None,
                   x_rescaled=False,
                   y_rescaled=False):
    if ax is None:
        flag_show = True
        ax = plt.subplot(111)
    else:
        flag_show = False
    if not isinstance(Lys, list):
        print("Lys should be a list")
        sys.exit()
    for Ly in Lys:
        q2, hq2 = read(Lx, Ly)
        if y_rescaled:
            y = hq2[sigma_y] * Ly
        else:
            y = hq2[sigma_y]
        if x_rescaled:
            x = q2 * Ly
        else:
            x = q2
        ax.plot(x, y, "o", label="$L_y=%d$" % Ly, alpha=0.8, ms=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    if y_rescaled:
        ylabel = r"$L_y \cdot \langle |h_q|^2\rangle_t$"
    else:
        ylabel = r"$\langle |h_q|^2\rangle_t$"
    if x_rescaled:
        xlabel = r"$L_y \cdot (q / 2\pi)^2$"
    else:
        xlabel = r"$(q / 2\pi)^2$"
    ax.set_ylabel(ylabel, fontsize="large")
    ax.set_xlabel(xlabel, fontsize="large")
    ax.legend()
    if flag_show:
        plt.show()
        plt.close()


def two_panel_varied_Ly(Lx, Lys, sigma_y, show=True):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    plot_varied_Ly(Lx, sigma_y, Lys, ax=ax1)
    plot_varied_Ly(Lx, sigma_y, Lys, ax=ax2, y_rescaled=True)
    llabel = r"${\rm slope}=-1$"
    add_line(ax1, 0, 0.9, 0.95, -1, scale="log", label=llabel, xl=0.3, yl=0.45)
    add_line(ax2, 0, 0.9, 0.95, -1, scale="log", label=llabel, xl=0.3, yl=0.45)
    plt.suptitle(
        r"$\eta=0.18, \epsilon=0, \rho_0=1, L_x=%d, \sigma_y=%d$" %
        (Lx, sigma_y),
        fontsize="x-large")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if show:
        plt.show()
    else:
        plt.savefig(r"data\hq\varied_Lx_%d_sy_%d.pdf" % (Lx, sigma_y))
    plt.close()


def plot_varied_Lx(Ly, sigma_y, Lxs, eta=0.18, eps=0, ax=None):
    if ax is None:
        flag_show = True
        ax = plt.subplot(111)
    else:
        flag_show = False
    if not isinstance(Lxs, list):
        print("Lxs should be a list")
        sys.exit()
    for Lx in Lxs:
        q2, hq2 = read(Lx, Ly)
        ax.plot(q2, hq2[sigma_y], "o", label="$L_x=%d$" % Lx, alpha=0.8, ms=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    if flag_show:
        plt.show()
        plt.close()


if __name__ == "__main__":
    # plot_varied_sigma_y(180, 1000)
    Lx = 180
    Lys = [200, 400, 600, 800, 1000]
    sigma_y = 10
    two_panel_varied_Ly(Lx, Lys, sigma_y, show=False)
    # plot_varied_sigma_y(180, 1000, save=True)
