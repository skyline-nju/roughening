import numpy as np
import matplotlib.pyplot as plt
import struct
# from scipy.ndimage import gaussian_filter
# from operator import itemgetter, attrgetter
import glob


def read(file):
    f = open(file, 'rb')
    buff = f.read()
    data = struct.unpack('%df' % (len(buff) // 4), buff)
    f.close()
    x, y, theta = np.array(data).reshape(len(buff) // 12, 3).T
    return x, y, theta


def hist(x, y, theta, ncols, nrows, Lx, Ly):
    rho_hist = np.zeros((nrows, ncols), int)
    # rho_hist_weighted = np.zeros((nrows, ncols))
    vx_hist = np.zeros((nrows, ncols))
    vy_hist = np.zeros((nrows, ncols))
    vx, vy = np.cos(theta), np.sin(theta)
    lx = Lx / ncols
    ly = Ly / nrows
    one_area = int(ncols / Lx * nrows / Ly)
    for k in range(x.size):
        col = int(x[k] / lx)
        row = int(y[k] / ly)
        rho_hist[row, col] += 1
        vx_hist[row, col] += vx[k]
        vy_hist[row, col] += vy[k]
    rho_hist *= one_area
    return rho_hist, vx_hist, vy_hist


class Chain:
    def __init__(self, x0, field, const_force):
        self.field = field * 2
        self.nRows, self.nCols = field.shape
        self.x = np.ones(self.nRows, int) * x0
        self.y = np.arange(self.nRows)
        self.tension = np.zeros(self.nRows, int)
        self.Fc = const_force
        self.motion_dict = {
            (-1, -1): (-1, 0, 0),
            (-1, 0): (-1, 0, 0),
            (-1, 1): (0, 0, 2),
            (0, -1): (-1, 0, 0),
            (0, 0): (-1, 0, 0),
            (0, 1): (0, 0, 2),
            (1, -1): (0, 2, 0),
            (1, 0): (0, 2, 0),
            (1, 1): (0, 1, 1)
        }

    def move_one_node(self, row, F):
        row_pre = row - 1
        row_next = row + 1
        if row == 0:
            row_pre = self.nRows - 1
        elif row == self.nRows - 1:
            row_next = 0
        dcol1 = self.x[row_pre] - self.x[row]
        dcol2 = self.x[row_next] - self.x[row]
        half_F = F // 2
        dx, dFpre, dFnext = self.motion_dict[(dcol1, dcol2)]
        self.x[row] += dx
        self.tension[row_pre] += dFpre * half_F
        self.tension[row_next] += dFnext * half_F
        if dx != 0:
            isMoved = True
        else:
            isMoved = False
        return isMoved

    def move_chain(self):
        self.tension = np.zeros(self.nRows, int)
        sorted_index = np.argsort(self.x)
        is_chain_move = False
        for row in sorted_index:
            col = self.x[row]
            col_pre = col - 1 if col >= 0 else self.nCols - 1
            F = self.field[row, col_pre] + self.Fc + self.tension[row]
            if F < 0:
                is_node_move = self.move_one_node(row, F)
                if is_node_move:
                    is_chain_move = True
        return is_chain_move

    def eval(self):
        while True:
            is_chain_move = self.move_chain()
            if not is_chain_move:
                break


if __name__ == "__main__":
    Lx, Ly = 200, 25600
    eta = 0.35
    eps = 0.02
    seed = 98000

    files = glob.glob('%.2f_%.2f_%d_%d_%d*.bin' % (eta, eps, Lx, Ly, seed))
    h0 = 150
    x, y, theta = read(files[0])
    lx = ly = 1
    nCols, nRows = int(Lx / lx), int(Ly / ly)
    box = [0, nRows, 0, nCols]
    rho, vx, vy = hist(x, y, theta, nCols, nRows, Lx, Ly)

    # module = np.sqrt(vx * vx + vy * vy)
    # mask = rho > 0
    # module[mask] /= rho[mask]

    rho[rho > 4] = 4
    chain = Chain(h0, rho, -4)
    chain.eval()
    plt.imshow(
        rho.T, origin='lower', aspect='auto', interpolation="none", extent=box)
    plt.plot(chain.y, chain.x, 'w-')
    plt.axis(box)

    plt.show()
    plt.close()

    # #rho[vx < 0] = 0
    # rho1 = rho.copy()
    # rho1[rho1 > 4] = 4
    # c1 = Chain(nCols - dh, rho1, -4)
    # c1.eval()

    # plt.imshow(
    #       rho1.T,
    #       origin = 'lower',
    #       aspect = 'auto',
    #       interpolation = 'none',
    #       extent = box)
    # plt.plot(c1.y, c1.x, 'k-')
    # plt.axis(box)
    # plt.colorbar()
    # plt.show()
    # plt.close()

    # mask = rho > 0
    # vxm = vx.copy()
    # vxm[mask] /= rho[mask]

    # vxm1 = np.zeros (nRows)
    # vxm2 = np.zeros (nRows)
    # kx = 50
    # ky = 5
    # for row in range(nRows):
    #     col = c1.x[row]
    #     vxm1[row] = np.mean(vxm[row, col - ky + 1 : col + 1])
    # for row in range(nRows):
    #     for i in range(row - kx, row + kx + 1):
    #         if i >= nRows:
    #             i -= nRows
    #         vxm2[row] += vxm1[i]
    #     vxm2[row] /= (2 * kx + 1)
    # vxm2 *= 200

    # plt.plot(c1.y, c1.x, 'b-', label = '$w^2=%.3f$'%(np.var(c1.x)))
    # plt.plot(c1.y, vxm2, 'r-')
    # plt.axis(box)
    # plt.legend()
    # plt.show()
    # plt.close()
