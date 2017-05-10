import numpy as np
import matplotlib.pyplot as plt
import struct
from w_Ly_short_time import add_line


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


class RSOS:
    def __init__(self, L, N, flag=False):
        self.L = L
        self.N = N
        self.h = np.zeros(L, int)
        self.flag = flag

    def deposit(self):
        while True:
            i = np.random.randint(self.L)
            dh = self.h[i] - self.h[i - 1] + 1
            if abs(dh) <= self.N:
                j = i + 1
                if j >= self.L:
                    j = 0
                dh = self.h[j] - self.h[i] - 1
                if abs(dh) <= self.N:
                    self.h[i] += 1
                    break
            if self.flag:
                break

    def eval(self, tmax, ts):
        w = np.zeros(len(ts))
        pos = 0
        for i in range(tmax):
            self.deposit()
            if i + 1 == ts[pos]:
                w[pos] = np.var(self.h)
                if pos < len(ts) - 1:
                    pos += 1
        return w


def short_time_regime():
    tmax = 10000000
    ts = get_log_time(tmax, 1.08)
    L = [256, 512, 1024, 2048, 4096]
    ws = np.zeros((5, len(ts)))
    for i, l in enumerate(L):
        rsos = RSOS(l, 1)
        ws[i] = rsos.eval(tmax, ts)
    np.savez("short_time.npz", t=np.array(ts), w=ws)


def read(file):
    with open(file, "rb") as f:
        buff = f.read()
        L = int(file.split("_")[1])
        buff_size = len(buff)
        data = struct.unpack("%di" % (buff_size // 4), buff)
        data = np.array(data).reshape((buff_size // 4 // L, L))
    return data


def cal_hq(hs):
    nrows, ncols = hs.shape
    A2 = np.zeros(ncols // 2 + 1)
    q = np.arange(ncols // 2 + 1) / ncols
    for h in hs:
        dh = h - np.mean(h)
        hq = np.fft.rfft(dh) / ncols
        A2 += np.abs(hq)**2
    A2 = A2 / nrows
    return q, A2


if __name__ == "__main__":
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(10, 4))
    Lxs = [256, 512, 1024, 2048, 4096]
    ws = np.zeros(len(Lxs))
    for i, Lx in enumerate(Lxs):
        data = read("RSOS_%d_1_1.bin" % Lx)[250:]
        q, A2 = cal_hq(data)
        ax3.loglog(q**2, A2 * Lx, "o", ms=1, label=r"$L=%d$" % Lx)
        w = np.array([np.var(h) for h in data])
        ws[i] = np.mean(w)
    ax3.set_xlabel(r"$(q/2\pi)^2$")
    ax3.set_ylabel(r"$L \cdot \langle |h_q|^2\rangle_t$")
    ax3.legend()

    ax1.loglog(Lxs, ws, "o")
    ax1.set_xlabel(r"$L$")
    ax1.set_ylabel(r"$\langle w^2\rangle_t$")

    data = np.load("short_time.npz")
    t = data["t"]
    ws = data["w"]
    for i, w in enumerate(ws):
        ax2.loglog(t, w, "o", ms=1, label=r"$L=%d$" % Lxs[i])
    ax2.set_xlabel(r"$t$")
    ax2.set_ylabel(r"$w^2$")
    ax2.legend()

    add_line(ax1, 0, 0.1, 1, 1, label="slope=1", scale="log")
    add_line(
        ax2, 0, 0.1, 1, 2 / 3, label="slope=2/3", xl=0.4, yl=0.6, scale="log")
    add_line(ax3, 0, 0.9, 1, -1, label="slope=-1", scale="log")
    plt.suptitle("Restricted Solid-on-Solid (RSOS) Model")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    # plt.show()
    plt.savefig("RSOS.pdf")
    plt.close()
