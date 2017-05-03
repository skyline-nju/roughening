"""
Calculate 2D spatial correlation functions of density, current and orientation
fileds.
"""

import numpy as np


def corr2d(u, v=None):
    def autoCorr2d(f):
        F = np.fft.rfft2(f)
        corr = np.fft.irfft2(F * F.conj())
        return corr

    if v is None:
        res = autoCorr2d(u) / u.size
    else:
        res = (autoCorr2d(u) + autoCorr2d(v)) / u.size
    res = np.fft.fftshift(res)
    return res


def cal_corr(vx, vy, num, dA):
    mask = num > 0
    Jx = vx / dA
    Jy = vy / dA
    rho = num / dA
    module = np.zeros_like(Jx)
    module[mask] = np.sqrt(Jx[mask]**2 + Jy[mask]**2)
    ux = np.zeros_like(Jx)
    uy = np.zeros_like(Jy)
    ux[mask] = Jx[mask] / module[mask]
    uy[mask] = Jy[mask] / module[mask]
    valid_count = corr2d(mask)
    corr_rho = corr2d(rho)
    corr_J = corr2d(Jx, Jy) / valid_count
    corr_u = corr2d(ux, uy) / valid_count
    return corr_rho, corr_J, corr_u


class Corr2D:
    def __init__(self, file, dA):
        self.outfile = "corr_" + file.replace(".bin", ".npz")
        self.dA = dA
        self.count = 0

    def accu(self, vx, vy, num):
        if self.count == 0:
            self.C_rho, self.C_J, self.C_u = cal_corr(vx, vy, num, self.dA)
        else:
            corr_rho, corr_J, corr_u = cal_corr(vx, vy, num, self.dA)
            self.C_rho += corr_rho
            self.C_J += corr_J
            self.C_u += corr_u
        self.count += 1

    def output(self):
        C_rho = self.C_rho / self.count
        C_J = self.C_J / self.count
        C_u = self.C_u / self.count
        np.savez(self.outfile, C_rho=C_rho, C_J=C_J, C_u=C_u)


if __name__ == "__main__":
    import load_snap
    import os
    import sys

    file = sys.argv[1]
    path, file = file.split("/")
    os.chdir(path)
    str_list = file.split("_")
    Lx = int(str_list[3])
    Ly = int(str_list[4])
    snap = load_snap.RawSnap(file)
    corr = Corr2D(file, 1)
    for frame in snap.gene_frames(beg_idx=300):
        x, y, theta = frame
        rho, vx, vy = load_snap.coarse_grain(x, y, theta, Lx, Ly)
        corr.accu(vx, vy, rho)
    corr.output()
