import os
import sys
import numpy as np
import platform
import matplotlib
import load_snap
if platform.system() is "Windows":
    import matplotlib.pyplot as plt
else:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt


def solve(z, zc, left, right):
    root = []
    for i in range(right - 1, left - 1, -1):
        if z[i - 1] > zc and z[i] <= zc:
            root.append(i)
    return root


def crosspoint(x1, x2, y1, y2, y, Lx):
    x = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
    if x < 0:
        x += Lx
    return x


def coarse_grain(z0, nrows, ncols):
    nrows0, ncols0 = z0.shape
    drows = nrows0 // nrows
    dcols = ncols0 // ncols
    z = np.zeros((nrows, ncols))
    for j in range(nrows):
        for i in range(ncols):
            j1 = j * drows
            j2 = (j + 1) * drows
            i1 = i * dcols
            i2 = (i + 1) * dcols
            z[j, i] = np.mean(z0[j1:j2, i1:i2])
    return z


def ini(rho, rowM, colM, isovalue, x, Lx):
    colM2 = colM // 10
    rho2 = coarse_grain(rho, rowM, colM2)

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
        # print("failed to initialize")
        plt.contourf(rho2)
        plt.colorbar()
        plt.show()
        plt.close()
    else:
        # print('Initializition:', len((ini_col)))
        pass

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
    dx = Lx / colM
    xm = crosspoint(x[index_m] - dx, x[index_m], rhox[index_m - 1],
                    rhox[index_m], isovalueNew, Lx)
    return ini_row, ini_col, xm


def isoline(rho0, Lx, Ly, colM, rowM, isovalue=2):
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
                        # print("failed to find iso_col[%d]" % row)
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

    rho = coarse_grain(rho0, rowM, colM)
    dx, dy = Lx / colM, Ly / rowM
    x = np.linspace(0.5 * dx, Lx - 0.5 * dx, colM)
    y = np.linspace(0.5 * dy, Ly - 0.5 * dy, rowM)
    ini_row, ini_col, xm = ini(rho, rowM, colM, isovalue, x, Lx)
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
                    row, col], isovalue, Lx))
    for i in range(len(h)):
        if h[i] - xm > Lx * 0.5:
            h[i] -= Lx
        elif h[i] - xm < -Lx * 0.5:
            h[i] += Lx
    h = np.array(h)
    mean = np.mean(h)
    var = np.var(h)
    return var, mean, xm, yc, h


if __name__ == "__main__":
    if platform.system() is "Windows":
        os.chdir(r"D:\tmp")
        Lx = 220
        Ly = 25600
        N = Lx * Ly
        seed = 1234
        file = r"cB_0.35_0.02_%d_%d_%d_%d_%d_1.06_%d.bin" % (Lx, Ly, Lx, Ly, N,
                                                             seed)
        snap = load_snap.CoarseGrainSnap(file)
    else:
        file = sys.argv[1]
        path, file = file.split("/")
        os.chdir(path)
        snap = load_snap.CoarseGrainSnap(file)
        para_list = file.replace(".bin", "").split("_")
        Lx = int(para_list[3])
        Ly = int(para_list[4])
        seed = int(para_list[9])

    outfile = "old_%d_%d_%d.dat" % (Lx, Ly, seed)
    f = open(outfile, "w")
    for frame in snap.gene_frames():
        t, vx, vy, rho = frame
    f.close()
