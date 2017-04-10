""" Modified snakes to detect the surface of bands.

    Inspired by the originally simple implementation of snakes, see
    http://www.crisluengo.net/index.php/archives/217
"""

import numpy as np
import numpy.matlib as npmat
import matplotlib.pyplot as plt
import scipy.ndimage


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


def get_ext_force(img, sigma):
    """ Get external force from the image.

        Parameters:
        --------
        img: np.ndarray
            A array with shape n by m.

        Returns:
        --------
        fx: np.ndarray
            External force in the x direction.
        fy: np.ndarray
            External force in the y direction.

    """
    smoothed = scipy.ndimage.gaussian_filter(img, sigma)
    grady, gradx = np.gradient(smoothed)

    def fx(x, y):
        return gradx[y.round().astype(int), x.round().astype(int)]

    def fy(x, y):
        return grady[y.round().astype(int), x.round().astype(int)]

    return fx, fy


def one_step(x, y, gamma, invP, fx, fy):
    x_new = np.dot(invP, x + gamma * fx(x, y))
    y_new = np.dot(invP, y + gamma * fy(x, y))
    x, y = x_new.A1, y_new.A1
    return x, y


def create_demo(n=200):
    img = np.random.randn(n, n)
    img -= img.min()
    for i, y in enumerate(np.linspace(0, 1, n)):
        for j, x in enumerate(np.linspace(0, 1, n)):
            if (x - 0.5)**2 + (y - 0.5)**2 < 0.1:
                img[i, j] += 10

    t = np.linspace(0, 2 * np.pi, 100)
    x = 80 * np.cos(t) + 100
    y = 80 * np.sin(t) + 100
    return img, x, y


def get_fx(rho, rho_t):
    drho = rho - rho_t

    def fx(x, y):
        return drho[y, x.round().astype(int)]
    return fx


def one_step_x(x, y, gamma, invP, fx):
    x_new = np.dot(invP, x + gamma * fx(x, y)).A1
    delta_x = np.mean((x_new - x)**2)
    x = x_new
    return x, delta_x


def find_interface(rho, alpha=0.1, beta=0.5, gamma=0.5, count=1000, rho_t=2):
    """ Find the interface of band.

        Remaining Issues:
            1) Need consider periodic boundary condition in the x direction.
        Besides, need give suitable value of x0 to initialize x.
            2) When the number of rows is too large, there will be memory
        error since the matrix is too large.
            3) At present, y is fixed. Maybe there will be better result if y
        is free to move.

        Parameters:
        --------
        rho: np.ndarray
            Smoothed density fileds.
        rho_t: float
            Threshold value of density.
        alpha: float
            Control the length of the interface.
        beta: float
            Control the smoothness of the interface.
        gamma: float
            The step size the points move.
        count: int
            The number of total steps.

        Returns:
        x: np.ndarray
            The x coordination of the interface.
    """
    nrows, ncols = rho.shape
    y = np.arange(nrows)
    x = np.ones(nrows) * 140
    invP = get_inverse_mat(alpha, beta, gamma, y.size)
    drho = rho - rho_t
    for i in range(count):
        x_new = np.dot(invP, x + gamma * drho[y, x.round().astype(int)]).A1
        x = x_new
    return x


if __name__ == "__main__":
    import os
    import load_snap
    os.chdir(r"D:\tmp")
    Lx = 150
    Ly = 900
    snap = load_snap.RawSnap(r"so_0.35_0_%d_%d_%d_2000_1234.bin" %
                             (Lx, Ly, Lx * Ly))
    x, y, theta = snap.read_frame(130)
    rho = load_snap.coarse_grain(x, y)
    rho_smoothed = scipy.ndimage.gaussian_filter(rho, sigma=[1.5, 0.8])

    rho0 = rho_smoothed.copy()
    rho0[rho0 < 2] = 0

    y = np.linspace(0.5, Ly-0.5, Ly)
    alphas = np.arange(5)
    betas = np.arange(4)
    wdt = np.zeros((alphas.size, betas.size))
    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            x = find_interface(rho_smoothed, alpha=alpha, beta=beta, count=2000)
            wdt[i, j] = np.std(x)
    plt.contourf(betas, alphas, wdt)
    plt.colorbar()
    plt.show()
    plt.close()