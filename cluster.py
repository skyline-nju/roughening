""" DBSCAN

    Density-Based Spatial Clustering of Applications with Noise.
"""

import os
import numpy as np
import matplotlib.pyplot as plt


def DBSCAN(x, y, r0, MinPts, Lx, Ly, cell_list):
    n = x.size
    visited = np.zeros(n, bool)
    clustered = np.zeros(n, bool)
    c = []
    for i in range(x.size):
        if visited[i]:
            continue
        visited[i] = True
        neighborPts = region_query(i, x, y, r0, Lx, Ly, cell_list)
        if len(neighborPts) < MinPts:
            pass
        else:
            c_new = expand_cluster(i, neighborPts, visited, clustered, x, y,
                                   r0, MinPts, Lx, Ly, cell_list)
            c.append(c_new)
    return c


def expand_cluster(i, neighborPts, visited, clustered, x, y, r0, MinPts, Lx,
                   Ly, cell_list):
    c_new = [i]
    clustered[i] = True
    j = 0
    while j < len(neighborPts):
        k = neighborPts[j]
        if not visited[k]:
            visited[k] = True
            neighborPts_new = region_query(k, x, y, r0, Lx, Ly, cell_list)
            if len(neighborPts_new) >= MinPts:
                neighborPts.extend(neighborPts_new)
        if not clustered[k]:
            clustered[k] = True
            c_new.append(k)
        j += 1
    return c_new


def region_query(i, x, y, r0, Lx, Ly, cell_list):
    def cal_dis(i, j):
        dx = x[j] - x0
        dy = y[j] - y0
        if dx < -0.5 * Lx:
            dx += Lx
        elif dx > 0.5 * Lx:
            dx -= Lx
        if dy < -0.5 * Ly:
            dy += Ly
        elif dy > 0.5 * Ly:
            dy -= Ly
        return dx ** 2 + dy ** 2

    neighbor = []
    x0 = x[i]
    y0 = y[i]
    col0 = int(x0)
    row0 = int(y0)
    for row in range(row0-1, row0+2):
        if row < 0:
            row += Ly
        elif row >= Ly:
            row -= Ly
        for col in range(col0-1, col0+2):
            if col < 0:
                col += Lx
            elif col >= Lx:
                col -= Lx
            if cell_list[row, col] is not 0:
                for j in cell_list[row, col]:
                    if j != i and cal_dis(i, j) < r0:
                        neighbor.append(j)
    return neighbor


def create_cell_list(x, y, Lx, Ly):
    cell = np.zeros((Ly, Lx), list)
    n = x.size
    for i in range(n):
        col = int(x[i])
        row = int(y[i])
        if cell[row, col] == 0:
            cell[row, col] = [i]
        else:
            cell[row, col].append(i)
    return cell


def show_cluster(x, y, c, ax):
    c = sorted(c, key=lambda x: len(x), reverse=True)
    clist = plt.cm.jet(np.linspace(0, 1, 11))
    for i, ci in enumerate(c):
        if i < 10:
            color = clist[i]
        else:
            color = clist[-1]
        ax.plot(x[ci], y[ci], "o", c=color, ms=0.5)


if __name__ == "__main__":
    import load_snap
    os.chdir(r"D:\code\VM\VM\snap")
    Lx = 180
    Ly = 20
    for t in range(10000, 100000, 1000):
        file = r"s_0.35_0_1_%d_%d_312_%08d.bin" % (Lx, Ly, t)
        snap = load_snap.RawSnap(file)
        for frame in snap.gene_frames():
            x, y, theta = frame
            # mask = x < 40
            # x = x[mask]
            # y = y[mask]
            # theta = theta[mask]

            # mask = y < 40
            # x = x[mask]
            # y = y[mask]
            # theta = theta[mask]

            plt.subplot(121)
            # plt.plot(x, y, 'o', ms=1)
            plt.scatter(y, x, c=theta, s=1, cmap="hsv")
            ax = plt.subplot(122)
            cell_list = create_cell_list(x, y, Lx, Ly)
            c = DBSCAN(x, y, 1, 3, Lx, Ly, cell_list)
            print(len(c))
            show_cluster(y, x, c, ax)
            plt.show()
            plt.close()