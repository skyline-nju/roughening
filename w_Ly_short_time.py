""" Plot width against t in log-log scales for the short time regime. """

import numpy as np
import matplotlib.pyplot as plt
import os
import glob


def read(file):
    with open(file) as f:
        lines = f.readlines()
        t = np.zeros(len(lines), int)
        w = np.zeros(t.size)
        for i, line in enumerate(lines):
            s = line.split("\t")
            t[i] = int(s[0])
            w[i] = float(s[1])
    return t, w


def one_panel(eps, Ly, ax=None, ms=4):
    if ax is None:
        flag_show = True
        ax = plt.subplot(111)
    else:
        flag_show = False

    files = glob.glob("%.2f_%05d_*.dat" % (eps, Ly))
    for file in files:
        t, w = read(file)
        ax.plot(t, w, "o", ms=ms)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid("on")

    if flag_show:
        plt.show()
        plt.close()


def add_line(ax,
             x_beg,
             y_beg,
             x_end,
             slope,
             label=None,
             xl=None,
             yl=None,
             fontsize="x-large",
             scale="lin",
             c="#7f7f7f"):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    if scale == "lin":
        slope_new = slope * (xmax - xmin) / (ymax - ymin)
    else:
        slope_new = slope * (np.log10(xmax / xmin) / np.log10(ymax / ymin))
    x = np.linspace(x_beg, x_end, 100)
    y = slope_new * (x - x_beg) + y_beg
    ax.plot(x, y, "-.", transform=ax.transAxes, color=c)
    if label is not None:
        width = ax.bbox.width
        height = ax.bbox.height
        deg = np.arctan(slope_new * height / width) * 180 / np.pi
        dx = x_end - x_beg
        if xl is None:
            xl = x_beg + dx * 0.3
        if yl is None:
            yl = y_beg + dx * 0.6 * slope_new
        ax.text(
            xl,
            yl,
            label,
            transform=ax.transAxes,
            rotation=deg,
            color=c,
            fontsize=fontsize)


if __name__ == "__main__":
    os.chdir(r"data/short_time_old")
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 6))
    eps_list = [0, 0.02]
    Ly_list = [6400, 12800, 25600]
    for row, eps in enumerate(eps_list):
        for col, Ly in enumerate(Ly_list):
            one_panel(eps, Ly, axes[row][col], ms=1)
            # set xlim, ylim
            if row == 0:
                axes[row][col].set_ylim(0.5, 50)
            else:
                axes[row][col].set_ylim(1, 1000)
            axes[row][col].set_xlim(100, 1000000)

            # set title
            if col == 0:
                axes[row][0].set_ylabel(
                    r"$\epsilon=%g$" % eps, fontsize="x-large")
            if row == 0:
                axes[0][col].set_title(r"$L_y=%d$" % Ly, fontsize="x-large")

            # set label
            tf = axes[row][col].transAxes
            axes[row][col].text(
                0.02, 0.9, r"$w^2$", fontsize="x-large", transform=tf)
            axes[row][col].text(
                0.95, 0.02, r"$t$", fontsize="x-large", transform=tf)

    add_line(axes[0][0], 0.20, 0.12, 0.6, 0.5, scale="log", label="slope=1/2")
    add_line(axes[0][1], 0.18, 0.14, 0.6, 0.5, scale="log", label="slope=1/2")
    add_line(axes[0][2], 0.18, 0.14, 0.6, 0.5, scale="log", label="slope=1/2")
    add_line(
        axes[1][0], 0.18, 0.03, 0.6, 2 / 3, scale="log", label="slope=2/3")
    add_line(
        axes[1][1], 0.18, 0.05, 0.6, 2 / 3, scale="log", label="slope=2/3")
    add_line(
        axes[1][2], 0.18, 0.05, 0.6, 2 / 3, scale="log", label="slope=2/3")

    plt.tight_layout()
    plt.show()
    plt.close()
