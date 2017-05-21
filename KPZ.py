import numpy as np
import matplotlib.pyplot as plt
import struct
from RSOS import run
from skewness import skew, kurt
# from w_Ly_short_time import add_line


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
    def __init__(self, L, N):
        self.L = L
        self.N = N
        self.h = np.zeros(L, int)

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

    def eval(self, tmax, ts):
        ht = np.zeros((len(ts), self.L))
        pos = 0
        for i in range(tmax):
            self.deposit()
            if i + 1 == ts[pos]:
                ht[pos] = self.h
                if pos < len(ts) - 1:
                    pos += 1
        return ht


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
    tmax = 1000000000
    ts = get_log_time(tmax, 1.08)
    ts = np.array(ts)
    n = ts.size
    L = 1024
    gamma1 = np.zeros(n)
    gamma2 = np.zeros(n)
    sample_size = 1
    for seed in range(10, sample_size + 10):
        ht = np.zeros(ts.size * L, int)
        run(L, 1, seed, ts, ht)
        ht = ht.reshape(n, L)
        gamma1 += np.array([skew(i) for i in ht])
        gamma2 += np.array([kurt(i) for i in ht])
    gamma1 /= sample_size
    gamma2 /= sample_size

    plt.subplot(211)
    plt.plot(ts, gamma1, "o")
    plt.xscale("log")
    plt.xlim(100)
    plt.ylim(-1, 1.5)
    plt.ylabel(r"$\gamma_1$")
    plt.axhline(0.29)
    plt.subplot(212)
    plt.plot(ts, gamma2, "o")
    plt.xscale("log")
    plt.xlim(100)
    plt.ylim(-1.5, 1.5)
    plt.xlabel(r"$t$")
    plt.ylabel(r"$\gamma_2$")
    plt.axhline(0.16)
    plt.suptitle("RSOS model with L=1024")
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()
