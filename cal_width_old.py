import numpy as np
import matplotlib.pyplot as plt
import glob
import struct
# import sys


def read(file):
    s = file.replace(".bin", '').split('_')
    Lx, Ly = int(s[3]), int(s[4])
    eta, eps, Lx, Ly, seed, time = float(s[1]), float(s[2]), int(s[3]), int(
        s[4]), int(s[5]), int(s[6])
    f = open(file, 'rb')
    buff = f.read()
    f.close()
    n = len(buff)
    buff = struct.unpack('%dc' % n, buff)
    data = np.array([ord(i) for i in buff])
    rho = data.reshape(Ly // 10, Lx) * 0.1
    return rho, eta, eps, Lx, Ly, seed, time


def isoline(rho, Lx, Ly, colM, rowM):
    def solve(z, zc, left, right):
        root = []
        for i in range(right - 1, left - 1, -1):
            if z[i - 1] > zc and z[i] <= zc:
                root.append(i)
        return root

    def ini():
        colM2 = colM // 10
        rho2 = np.zeros((rowM, colM2))
        for j in range(rowM):
            for i in range(colM2):
                rho2[j, i] = np.mean(rho[j, i * 10:i * 10 + 10])

        ini_row, ini_col = [], []
        for row in range(rowM):
            root2 = solve(rho2[row, :], isovalue, 0, colM2)
            if len(root2) == 1:
                col2 = root2[0]
                left = col2 * 10 - 10
                right = col2 * 10 + 10
                root1 = solve(rho[row, :], isovalue, left, right)
                if len(root1) == 1:
                    col = root1[0]
                    begin = col + 2
                    end = col + 40
                    if rho[row, col - 2] > rho[row, col - 1] and rho[
                            row, col + 1] < rho[row, col]:
                        flag = True
                        for i in range(begin, end):
                            if i >= colM:
                                i -= colM
                            if rho[row, i] > isovalue:
                                flag = False
                                break
                        if flag:
                            ini_row.append(row)
                            ini_col.append(col)
        for j in range(rowM // 20):
            min_index = ini_col.index(min(ini_col))
            ini_col.pop(min_index)
            ini_row.pop(min_index)
        if len(ini_col) == 0:
            print("failed to initialize")
            plt.contourf(rho2)
            plt.colorbar()
            plt.show()
            plt.close()
        else:
            print('Initializition:', len((ini_col)))

        rhox = np.array([np.mean(rho[:, i]) for i in range(colM)])
        root = solve(rhox, isovalue, 0, colM)
        isovalueNew = isovalue
        if len(root) == 0:
            isovalueNew = isovalue * 0.75
            root = solve(rhox, isovalueNew, 0, colM)
        if len(root) == 0:
            isovalueNew = isovalue * 0.5
            root = solve(rhox, isovalueNew, 0, colM)
        index_m = root[0]
        xm = crosspoint(x[index_m] - dx, x[index_m], rhox[index_m - 1],
                        rhox[index_m], isovalueNew)
        return ini_row, ini_col, xm

    def cal_isoLine():
        def search(row, col, index, count=0, maxstep=3):
            if col >= colM:
                col -= colM
            if rho[row, col] <= isovalue and rho[row, col - 1] > isovalue:
                flag = True
                for i in range(col + 1, col + 4):
                    if i >= colM:
                        i -= colM
                    if (rho[row, i] > isovalue):
                        flag = False
                        break
                for i in range(col - 1, col - 4):
                    if (rho[row, i] < isovalue):
                        flag = False
                        break
                if flag:
                    return col
            else:
                if count >= maxstep:
                    return None
                else:
                    return search(row, col + index, index, count + 1)

        def find(row0, col0, d):
            pre_col = col0
            if d == 1:
                rowList = range(row0 + 1, row0 + rowM)
            else:
                rowList = range(row0 - 1, row0 - rowM, -1)
            for row in rowList:
                new_col = None
                if row >= rowM:
                    row -= rowM
                if (rho[row, pre_col] <= isovalue and
                        rho[row, pre_col - 1] > isovalue):
                    new_col = pre_col
                elif rho[row, pre_col - 1] < isovalue:
                    new_col = search(row, pre_col - 1, -1)
                    if new_col is None:
                        new_col = search(row, pre_col + 1, 1)
                else:
                    new_col = search(row, pre_col + 1, 1)
                    if new_col is None:
                        new_col = search(row, pre_col - 1, -1)
                if new_col is not None:
                    col1 = new_col
                    if row in ini_row and col1 != ini_col[ini_row.index(row)]:
                        col2 = ini_col[ini_row.index(row)]
                        if abs(col1 - pre_col) > abs(col2 - pre_col):
                            col1 = col2
                    if iso_col[row] is not None and iso_col[row] != col1:
                        col2 = iso_col[row]
                        if abs(col1 - pre_col) > abs(col2 - pre_col):
                            col1 = col2
                    pre_col = iso_col[row] = col1
                else:
                    col1, col2 = None, None
                    if row in ini_row:
                        col1 = ini_col[ini_row.index(row)]
                    if iso_col[row] is None:
                        col2 = iso_col[row]
                    if col1 is not None and col2 is not None:
                        if abs(col1 - pre_col) > abs(col2 - pre_col):
                            col1 = col2
                        pre_col = iso_col[row] = col1
                    elif col1 is not None:
                        pre_col = iso_col[row] = col1
                    elif col2 is not None:
                        pre_col = iso_col[row] = col2
                    else:
                        print("failed to find iso_col[%d]" % row)
                        # show2()
                        return False
            return True

        for i in range(len(ini_row)):
            if iso_col[ini_row[i]] is None:
                iso_col[ini_row[i]] = ini_col[i]
                success = find(ini_row[i], ini_col[i], 1)
                success = find(ini_row[i], ini_col[i], -1)
                if success:
                    break

    def crosspoint(x1, x2, y1, y2, y):
        x = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
        if x < 0:
            x += Lx
        return x

    def show():
        plt.subplot(121)
        plt.contourf(x, y, rho, [0, 1, 2, 3, 4, 5], extend='max')
        xc = []
        for i in h:
            if i > Lx:
                i -= Lx
            elif i < 0:
                i += Lx
            xc.append(i)
        plt.plot(xc, yc, 'k.-')
        for i in range(len(ini_col)):
            a = x[ini_col[i]]
            b = y[ini_row[i]]
            plt.plot(a, b, 'ro')
        plt.colorbar()
        plt.subplot(122)
        plt.plot(xc, yc, 'r.-')
        plt.axis([0, Lx, 0, Ly])
        plt.suptitle(r"$missed point: %d,\ w^2 =%f,\ <h> = %f$" %
                     (rowM - len(h), var, mean))
        plt.show()

    def show2():
        plt.contourf(rho, [0, 1, 2, 3, 4, 5, 6])
        plt.plot(ini_col, ini_row, 'ro')
        for row in range(rowM):
            if iso_col[row] is not None:
                plt.plot(iso_col[row], row, 'ks')
        plt.colorbar()
        plt.show()

    isovalue = 1.5
    dx, dy = Lx / colM, Ly / rowM
    x = np.linspace(0.5 * dx, Lx - 0.5 * dx, colM)
    y = np.linspace(0.5 * dy, Ly - 0.5 * dy, rowM)
    ini_row, ini_col, xm = ini()
    iso_col = [None] * rowM
    cal_isoLine()
    # h=np.zeros(rowM)
    h, yc = [], []
    for row in range(rowM):
        if iso_col[row] is not None:
            col = iso_col[row]
            yc.append(y[row])
            h.append(
                crosspoint(x[col] - dx, x[col], rho[row, col - 1], rho[
                    row, col], isovalue))
    for i in range(len(h)):
        if h[i] - xm > Lx * 0.5:
            h[i] -= Lx
        elif h[i] - xm < -Lx * 0.5:
            h[i] += Lx
    h = np.array(h)
    mean = np.mean(h)
    var = np.var(h)
    # if var>8 or rowM-len(h)>10: show()
    show()
    return var, mean, xm, yc, h


def get_var(eps, Ly, seed):
    Lx = 200
    mx = 100
    my = Ly // 20
    files = glob.glob('hist\\h*_%.2f_200_%05d_%d_*.bin' % (eps, Ly, seed))
    var = []
    t = []
    h = []
    xm = []
    # hs = []
    f = open('%.2f_%05d_%d.dat' % (eps, Ly, seed))
    lines = f.readlines()
    f.close()
    w0 = np.array([float(i.split('\t')[1]) for i in lines])
    h0 = np.array([float(i.split('\t')[2]) for i in lines])
    delta = np.zeros(len(h0))
    for i in range(1, len(h0)):
        if h0[i] < h0[i - 1]:
            delta[i] = delta[i - 1] + Lx
        else:
            delta[i] = delta[i - 1]
    begin, end = 3175, 3184
    wmin = min(w0[begin:end])
    wmax = max(w0[begin:end])
    dw = (wmax - wmin) / 210
    plt.figure(figsize=(14, 8))
    for count in range(begin, end):
        # if count%20!=0:
        #    continue
        ratio = 1
        rho0, eta, eps, Lx, Ly, seed, time = read(files[count])
        t.append(time)
        rho = np.zeros((my, mx))
        a = Lx // mx
        b = Ly // my // 10
        for j in range(my):
            for i in range(mx):
                rho[j, i] = np.mean(rho0[b * j:b * j + b, a * i:a * i + a])
        var_t, h_t, xm_t, yc, hlist = isoline(rho, Lx, Ly, mx, my)
        print("Ly=%d\tt=%d\tvar=%f" % (Ly, time, var_t))
        var.append(var_t)
        h.append(h_t)

        hlist += delta[count]
        hnew = h_t + delta[count]
        xm.append(xm_t)

        hlist = (hlist - hnew) * ratio + hnew
        index = int((var_t - wmin) / dw)
        if var_t < wmin:
            index = 0
        plt.plot(hlist, yc, color=plt.cm.gist_heat(index))
    plt.xlabel('h(y,t)', fontsize=20)
    plt.ylabel('y', fontsize=20)
    plt.suptitle(
        r'$\epsilon=%.2f,\ t\in (%d,%d),\ \Delta t=%d$' %
        (eps, begin * 100, end * 100, 2000),
        fontsize=20)
    plt.show()
    mean = sum(var) / len(var)
    # show()
    return t, mean, var, h, xm


if __name__ == "__main__":
    eps, Ly, seed = 0.02, 12800, 2560015
    t0, mean0, var0, h0, xm0 = get_var(eps, Ly, seed)
