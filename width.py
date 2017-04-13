import numpy as np
import matplotlib.pyplot as plt


def read(file):
    with open(file) as f:
        lines = f.readlines()
        t1 = []
        w1 = []
        t2 = []
        w2 = []
        for i, line in enumerate(lines):
            s = line.replace("\n", "").split("\t")
            t0 = int(s[0])
            w1_0 = float(s[1])
            w2_0 = float(s[2])
            if i > 0:
                if w1_0 - w1[-1] < 100:
                    t1.append(t0)
                    w1.append(w1_0)
                if w2_0 - w2[-1] < 100:
                    t2.append(t0)
                    w2.append(w2_0)
            else:
                t1.append(t0)
                t2.append(t0)
                w1.append(w1_0)
                w2.append(w2_0)
    return np.array(t1), np.array(w1), np.array(t2), np.array(w2)


if __name__ == "__main__":
    Lx = 150
    Ly = 150
    Lys = [150, 200, 300, 400, 500]
    file = r"data\width\w_0.35_0_%d_%d_%d_2000_1234.dat" % (Lx, Ly, Lx * Ly)
    t1, w1,  t2, w2 = read(file)
