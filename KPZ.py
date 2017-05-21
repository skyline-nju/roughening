import numpy as np
import matplotlib.pyplot as plt
import struct
from skewness import skew, kurt
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
    rsos = RSOS(4096, 1)
    tmax = 10000000
    ts = get_log_time(tmax, 1.02)
    ht = rsos.eval(tmax, ts)
    print(ht.shape)

    w = np.zeros(len(ts))
    hm = np.zeros(len(ts))
    gamma1 = np.zeros(len(ts))
    for i in range(len(ts)):
        w[i] = np.var(ht[i])
        hm[i] = np.mean(ht[i])
        gamma1[i] = skew(ht[i])
    plt.plot(ts, gamma1, "o")
    plt.xscale("log")
    plt.show()
    plt.close()


