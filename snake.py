""" Modified snakes to detect the surface of bands.

    Inspired by the originally simple implementation of snakes, see
    http://www.crisluengo.net/index.php/archives/217
"""

import numpy as np
import numpy.matlib as npmat
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter, laplace
from scipy.interpolate import splev, splrep


def get_inverse_mat(alpha, beta, gamma, n):
    """ Get the inverse matrix.

        Define P = I - gamma * A, where A is a n-by-n matrix.
        Return the inverse of P.

        There will be memory error if n is very large. For large n,
        need use scipy.sparse module.
    """
    a = gamma * (2 * alpha + 6 * beta) + 1
    b = gamma * (-alpha - 4 * beta)
    c = gamma * beta
    P = npmat.eye(n, k=0) * a
    P += npmat.eye(n, k=1) * b
    P += npmat.eye(n, k=2) * c
    P += npmat.eye(n, k=n - 2) * c
    P += npmat.eye(n, k=n - 1) * b
    P += npmat.eye(n, k=-1) * b
    P += npmat.eye(n, k=-2) * c
    P += npmat.eye(n, k=-n + 2) * c
    P += npmat.eye(n, k=-n + 1) * b
    return P.getI()


def create_GVF(f, mu, count, dt):
    u = np.zeros_like(f)
    v = np.zeros_like(f)
    fy, fx = np.gradient(f)
    b = fx**2 + fy**2
    c1 = b * fx
    c2 = b * fy
    for i in range(count):
        Delta_u = laplace(u, mode="constant")
        Delta_v = laplace(v, mode="constant")
        u = (1 - b * dt) * u + (mu * Delta_u + c1) * dt
        v = (1 - b * dt) * v + (mu * Delta_v + c2) * dt
    return u, v


def create_GVF_force(img, mu, count, dt):
    grady, gradx = np.gradient(img)
    gm2 = gradx**2 + grady**2
    u, v = create_GVF(gm2, mu, count, dt)

    def fx(rows, cols):
        return u[rows, cols]

    def fy(rows, cols):
        return v[rows, cols]

    return fx, fy


def create_img_force(img, sigma=[1, 1]):
    grady, gradx = np.gradient(img)
    gm = gradx**2 + grady**2
    gms = gaussian_filter(gm, sigma=sigma)
    ggmy, ggmx = np.gradient(gms)

    def fx(rows, cols):
        return ggmx[rows, cols]

    def fy(rows, cols):
        return ggmy[rows, cols]

    return fx, fy


def untangle(x, L):
    """ Turn a periodic array into a continous one. """
    x_new = x.copy()
    phase = 0
    for i in range(1, x.size):
        dx = x[i] - x[i - 1]
        if dx > Lx * 0.5:
            phase -= Lx
        elif dx < -Lx * 0.5:
            phase += Lx
        x_new[i] = x[i] + phase
    return x_new


def find_max_slope(rho_x):
    idx_beg = np.argmax(rho_x)
    idx = idx_beg
    rho_small = 0.35
    while True:
        if rho_x[idx % rho_x.size] < rho_small:
            break
        else:
            idx += 1
        if idx > idx_beg + rho_x.size // 2:
            idx = idx_beg
            rho_small += 0.05

    x0 = np.linspace(idx_beg+0.5, idx-0.5, idx-idx_beg)
    y0 = rho_x[idx_beg: idx]
    spl = splrep(x0, y0)
    xi = np.linspace(x0[0], x0[-1], 100)
    yi = splev(xi, spl)
    yi2 = splev(xi, spl, der=1)
    idx_min = np.argmin(yi2)
    return xi[idx_min], yi[idx_min]


def ini_snake(rho, rho_t, mode="line", dx=10):
    def find_x(rho_x, rho_t, x_pre=None):
        if x_pre is None:
            idx_beg = rho_x.size - 1
        else:
            idx_beg = int(x_pre) + ncols // 4
        idx_end = idx_beg - ncols
        for i in range(idx_beg, idx_end, -1):
            r1 = rho_x[(i - 1) % ncols]
            r2 = rho_x[i % ncols]
            if r1 > rho_t >= r2:
                x_t = (rho_t - r1) / (r2 - r1) + i - 1
                plt.plot(rho_x)
                plt.plot(x_t, rho_t, "o")
                x0, rho0 = find_max_slope(rho_x)
                plt.plot(x0, rho0, "s")
                plt.show()
                plt.close()
                return x_t

    nrows, ncols = rho.shape
    y = np.linspace(0.5, rho.shape[0] - 0.5, rho.shape[0])
    if mode == "line":
        rho_x = np.mean(rho, axis=0)
        x = np.ones(rho.shape[0]) * find_x(rho_x, rho_t) + dx
    elif mode == "spline":
        dy = 50
        x0 = np.zeros(nrows // dy + 1)
        y0 = np.zeros_like(x0)

        rho_x = np.mean(rho, axis=0)
        xm = find_x(rho_x, rho_t)
        for i in range(x0.size - 1):
            rho_x = np.mean(rho[i * dy:(i + 1) * dy, :], axis=0)
            if i == 0:
                x0[i] = find_x(rho_x, rho_t, x_pre=xm)
            else:
                x0[i] = find_x(rho_x, rho_t, x_pre=x0[i - 1])
            y0[i] = dy / 2 + i * dy
        x0 += dx
        x0[-1] = x0[0]
        y0[-1] = x0.size * dy - dy / 2
        x0 = untangle(x0, ncols)
        spl = splrep(y0, x0, per=True)
        x = splev(y, spl)
    return x, y


def set_residual_y(idx_ymin, Ly, nrows, alpha, beta):
    """ set residual array due to the periodic condition in y direction"""
    y_res = np.zeros(nrows)
    y_res[idx_ymin - 2] = -beta * Ly
    y_res[idx_ymin - 1] = (alpha + 3 * beta) * Ly
    y_res[idx_ymin] = -(alpha + 3 * beta) * Ly
    y_res[(idx_ymin + 1) % nrows] = beta * Ly
    return y_res


def find_interface(rho,
                   alpha=0.5,
                   beta=1,
                   gamma=0.25,
                   count=500,
                   rho_t=2,
                   dx=10):
    nrows, ncols = rho.shape
    Ly = nrows
    Lx = ncols
    x, y = ini_snake(rho, rho_t, dx=dx, mode="spline")
    invP = get_inverse_mat(alpha, beta, gamma, nrows)
    idx_ymin = 0
    y_res = set_residual_y(idx_ymin, Ly, nrows, alpha, beta)
    drho = rho - rho_t
    for i in range(count):
        X = np.floor(x).astype(int) % ncols
        Y = np.floor(y).astype(int)

        if not (0 < y[idx_ymin] < y[idx_ymin - 1]):
            y_res = set_residual_y(y.argmin(), Ly, nrows, alpha, beta)
        x = np.dot(invP, x + gamma * drho[Y, X]).A1
        y = np.dot(invP, y + gamma * y_res).A1

        y[y < 0] += Ly
        y[y >= Ly] -= Ly
        # if (i % 200 == 0):
        #     if i == 0:
        #         plt.plot(y, untangle(x, Lx), "k")
        #     else:
        #         plt.plot(y, untangle(x, Lx))
    x[x < 0] += Lx
    x[x >= Lx] -= Lx
    return x, y


def original_snake(x, y, img, alpha, beta, gamma=0.25, count=500):
    nrows, ncols = img.shape
    Ly = nrows
    Lx = ncols
    invP = get_inverse_mat(alpha, beta, gamma, nrows)
    idx_ymin = 0
    y_res = set_residual_y(idx_ymin, Ly, nrows, alpha, beta)
    fx, fy = create_img_force(img, sigma=[1, 1])
    for i in range(count):
        X = np.floor(x).astype(int) % ncols
        Y = np.floor(y).astype(int)

        if not (0 < y[idx_ymin] < y[idx_ymin - 1]):
            y_res = set_residual_y(y.argmin(), Ly, nrows, alpha, beta)

        x = np.dot(invP, x + gamma * fx(Y, X)).A1
        y = np.dot(invP, y + gamma * (y_res + fy(Y, X))).A1
        y[y < 0] += Ly
        y[y >= Ly] -= Ly

    x[x < 0] += Lx
    x[x > Lx] -= Lx
    return x, y


if __name__ == "__main__":
    import os
    import load_snap
    os.chdir(r"D:\tmp")
    Lx = 220
    Ly = 800
    snap = load_snap.RawSnap(r"so_0.35_0.02_%d_%d_%d_2000_1234.bin" %
                             (Lx, Ly, Lx * Ly))
    width = []
    for i, frame in enumerate(snap.gene_frames(188, 189)):
        x, y, theta = frame
        num = load_snap.coarse_grain2(
            x, y, theta, Lx=Lx, Ly=Ly, ncols=Lx, nrows=Ly)
        rho = gaussian_filter(num * 1.0, sigma=[1, 1])

        rho[rho > 6] = 6
        xl, yl = find_interface(rho, alpha=5, rho_t=1.5, dx=5)
        spl = splrep(yl, xl)
        yi = np.linspace(0.5, Ly - 0.5, Ly)
        xi = untangle(splev(yi, spl), Lx)
        w = np.var(xi)
        width.append(w)
        print(i, w)
        plt.scatter(y, x, c=theta, s=0.5, cmap="hsv")
        plt.plot(yl, xl)
        plt.title(r"$w^2=%g$" % w)
        plt.show()
        plt.close()

    plt.plot(width, "-o")
    plt.show()
