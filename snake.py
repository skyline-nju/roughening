""" Modified snakes to detect the surface of bands.

    Inspired by the originally simple implementation of snakes, see
    http://www.crisluengo.net/index.php/archives/217
"""

import numpy as np
import numpy.matlib as npmat
from scipy.ndimage.filters import gaussian_filter, laplace
from scipy.interpolate import splev, splrep
import platform
import matplotlib
if platform.system() is not "Windows":
    matplotlib.use("Agg")


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
        if dx > L * 0.5:
            phase -= L
        elif dx < -L * 0.5:
            phase += L
        x_new[i] = x[i] + phase
    return x_new


def ini(rho, rho_h, mode="line", dx=5):
    pass


def ini_snake(rho, rho_t, mode="line", dx=10):
    def find_x(rho_x, rho_t, x_pre=None):
        if x_pre is None:
            idx_beg = np.argmax(rho_x) + 40
        else:
            idx_beg = int(x_pre) + 40
        idx_end = idx_beg - ncols
        for i in range(idx_beg, idx_end, -1):
            r1 = rho_x[(i - 1) % ncols]
            r2 = rho_x[i % ncols]
            if r1 > rho_t >= r2:
                x_t = (rho_t - r1) / (r2 - r1) + i - 1
                plt.plot(rho_x)
                plt.plot(x_t, rho_t, "o")
                plt.show()
                plt.close()
                return x_t

    nrows, ncols = rho.shape
    y = np.linspace(0.5, rho.shape[0] - 0.5, rho.shape[0])
    if mode == "line":
        rho_x = np.mean(rho, axis=0)
        x = np.ones(rho.shape[0]) * find_x(rho_x, rho_t.mean()) + dx
    elif mode == "spline":
        dy = 50
        x0 = np.zeros(nrows // dy + 1)
        y0 = np.zeros_like(x0)

        rho_x = np.mean(rho, axis=0)
        if isinstance(rho_t, np.ndarray):
            rho_tm = np.mean(rho_t)
        else:
            rho_tm = rho_t
        xm = find_x(rho_x, rho_tm)
        for i in range(x0.size - 1):
            rho_x = np.mean(rho[i * dy:(i + 1) * dy, :], axis=0)
            if isinstance(rho_t, np.ndarray):
                rho_tm = np.mean(rho_t[i * dy:(i + 1) * dy])
            else:
                rho_tm = rho_t
            if i == 0:
                x0[i] = find_x(rho_x, rho_tm, x_pre=xm)
            else:
                x0[i] = find_x(rho_x, rho_tm, x_pre=x0[i - 1])
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
                   dx=5):
    nrows, ncols = rho.shape
    Ly = nrows
    Lx = ncols
    if Ly <= 150:
        mode = "line"
    else:
        mode = "spline"
    x, y = ini_snake(rho, rho_t, dx=dx, mode=mode)
    invP = get_inverse_mat(alpha, beta, gamma, nrows)
    idx_ymin = 0
    y_res = set_residual_y(idx_ymin, Ly, nrows, alpha, beta)
    if not isinstance(rho_t, np.ndarray):
        drho = rho - rho_t
    else:
        drho = np.array([rho[i] - rho_t[i] for i in range(rho_t.size)])
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


def test(eta,
         eps,
         Lx,
         Ly,
         dt,
         seed,
         show=True,
         output=False,
         t_beg=0,
         t_end=None):
    if platform.system() is not "Windows":
        os.chdir(r"snap_one")
    else:
        os.chdir(r"D:\tmp")
    snap = load_snap.RawSnap(r"so_%g_%g_%d_%d_%d_%d_%d.bin" %
                             (eta, eps, Lx, Ly, Lx * Ly, dt, seed))
    width1 = []
    width2 = []
    for i, frame in enumerate(snap.gene_frames(t_beg, t_end)):
        x, y, theta = frame
        num = load_snap.coarse_grain2(
            x, y, theta, Lx=Lx, Ly=Ly, ncols=Lx, nrows=Ly)
        rho = num * 1.0
        xl, rho_t = half_rho.find_interface(rho, sigma=[15, 1])
        rho_s = gaussian_filter(rho, sigma=[1, 1], mode="wrap")
        xl2, yl2 = find_interface(rho_s, alpha=1, rho_t=rho_t)
        rho_s[rho_s > 4] = 4
        xl = untangle(xl, Lx)
        xl2 = untangle(xl2, Lx)
        yl = np.linspace(0, Ly - 0.5, Ly)
        plt.imshow(rho_s.T, origin="lower", interpolation="none")
        plt.plot(yl, xl, yl2, xl2)
        plt.show()
        plt.close()
        w1 = np.var(xl)
        w2 = np.var(xl2)
        print(i, w1, w2)
        width1.append(w1)
        width2.append(w2)
        plt.plot(y, x, "o", ms=1)
        plt.show()
        plt.close()
        plt.plot(rho_t)
        plt.show()
        plt.close()
    if show:
        plt.subplot(211)
        plt.plot(width1)
        plt.subplot(212)
        plt.plot(width2)
        plt.show()
        plt.close()
    if output:
        lines = [
            "%d\t%f\t%f\n" % ((i + 1 + t_beg) * 2000, width1[i], width2[i])
            for i in range(len(width1))
        ]
        file = r"w_%g_%g_%d_%d_%d_%d_%d.dat" % (eta, eps, Lx, Ly, Lx * Ly, dt,
                                                seed)
        with open(file, "w") as f:
            f.writelines(lines)


if __name__ == "__main__":
    import os
    import matplotlib.pyplot as plt
    import load_snap
    import half_rho
    # test(0.35, 0, 150, 150, 2000, 1234, False, False, 215, 216)
    if platform.system() is not "Windows":
        os.chdir(r"snap_one")
    else:
        os.chdir(r"D:\tmp")
    Lx = 150
    Ly = 150
    snap = load_snap.RawSnap(r"so_%g_%g_%d_%d_%d_%d_%d.bin" %
                             (0.35, 0, Lx, Ly, Lx * Ly, 2000, 1234))
    width = []
    debug = True
    t_beg = 224
    t_end = None
    for i, frame in enumerate(snap.gene_frames(t_beg, t_end)):
        x, y, theta = frame
        rho = load_snap.coarse_grain2(
            x, y, theta, Lx=Lx, Ly=Ly, ncols=Lx, nrows=Ly) * 1.0
        xh, rho_h = half_rho.find_interface(rho, sigma=[15, 1], debug=debug)
        yh = np.linspace(0.5, Ly - 0.5, Ly)

        w = np.var(untangle(xh, Lx))
        width.append(w)
        print(i + t_beg, w)
    plt.plot(width)
    plt.show()
