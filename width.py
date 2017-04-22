import numpy as np
import os
import matplotlib.pyplot as plt


def read(file):
    with open(file) as f:
        lines = f.readlines()
        t = np.zeros(len(lines), int)
        w1 = np.zeros(len(lines))
        w2 = np.zeros(len(lines))
        for i, line in enumerate(lines):
            s = line.replace("\n", "").split("\t")
            t[i] = int(s[0])
            w1[i] = float(s[1])
            w2[i] = float(s[2])
    return t, w1, w2


def read2(file):
    with open(file) as f:
        lines = f.readlines()
        nrows = len(lines)
        ncols = len(lines[0].split("\t")) - 1
        t = np.zeros(nrows, int)
        w = np.zeros((nrows, ncols))
        for i, line in enumerate(lines):
            str_list = line.replace("\n", "").split("\t")
            t[i] = int(str_list[0])
            for j, s in enumerate(str_list[1:]):
                w[i, j] = float(s)
    return t, w


def filtering(t0, w0, max_dw=100):
    t = []
    w = []
    for i in range(t0.size):
        if len(t) > 0:
            if w0[i] - w[-1] < max_dw and w0[i] > 0:
                t.append(t0[i])
                w.append(w0[i])
        else:
            t.append(t0[i])
            w.append(w0[i])
    return np.array(t), np.array(w)


def plot_all():
    os.chdir(r"data\width")
    Lx = 200
    Lys = [
        150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 700, 800, 900, 1000
    ]
    w1m = []
    w2m = []
    for Ly in Lys:
        file = r"w_0.35_0_%d_%d_%d_2000_1234.dat" % (Lx, Ly, Lx * Ly)
        t, w1, w2 = read(file)
        t1, w1f = filtering(t, w1, 50)
        t2, w2f = filtering(t, w2)
        w1m.append(w1f[250:].mean())
        w2m.append(w2f[250:].mean())
        print(Ly, w1m[-1], w2m[-1])
        if Ly == 150:
            plt.subplot(211)
            plt.plot(w1)
            plt.plot(w1f)
            plt.subplot(212)
            plt.plot(w2)
            plt.plot(w2f)
            plt.show()
            plt.close()
    # plt.subplot(121)
    plt.plot(Lys, w1m, "-o")
    # plt.subplot(122)
    plt.plot(Lys, w2m, "-s")
    plt.xscale("log")
    plt.yscale("log")
    plt.show()
    plt.close()


if __name__ == "__main__":
    # os.chdir(r"D:\tmp")
    # t, w = read2("wh_0.35_0_150_900_135000_2000_1234.dat")
    # for wdt in w.T:
    #     plt.plot(t, wdt)
    #     plt.show()
    #     plt.close()
    plot_all()
