""" Smooth Filter. """

import numpy as np


def gauss(x, sigma=1):
    return 1 / np.sqrt(2 * np.pi) / sigma * np.exp(-0.5 * (x / sigma)**2)


def gauss2d(x, y, sigma_x=1, sigma_y=None):
    if sigma_y is None:
        sigma_y = sigma_x
    return 1 / (2 * np.pi * sigma_x * sigma_y) * np.exp(-0.5 * (
        (x / sigma_x)**2 + (y / sigma_y)**2))


def gauss_weight(sigma_x, sigma_y=None):
    if sigma_y is None:
        sigma_y = sigma_x
    lw_x = int(sigma_x * 3 + 0.5)
    lw_y = int(sigma_y * 3 + 0.5)
    weight = np.zeros((2 * lw_y + 1, 2 * lw_x + 1))
    count = 0
    for drow in range(-lw_y, lw_y + 1):
        row = drow + lw_y
        for dcol in range(-lw_x, lw_x + 1):
            col = dcol + lw_x
            weight[row, col] = gauss2d(dcol, drow, sigma_x, sigma_y)
            count += weight[row, col]
    weight /= count
    return weight


def gauss_filter(z, sigma):
    if isinstance(sigma, list):
        if len(sigma) == 2:
            sigma_x, sigma_y = sigma
    else:
        sigma_x = sigma
        sigma_y = sigma_x
    weight = gauss_weight(sigma_x, sigma_y)
    lw_y = (weight.shape[0] - 1) // 2
    lw_x = (weight.shape[1] - 1) // 2
    z_new = np.zeros_like(z)
    nrows, ncols = z.shape
    for row_c in range(nrows):
        for col_c in range(ncols):
            for j, drow in enumerate(range(-lw_y, lw_y + 1)):
                row = row_c + drow
                if row >= nrows:
                    row -= nrows
                for i, dcol in enumerate(range(-lw_x, lw_x + 1)):
                    col = col_c + dcol
                    if col >= ncols:
                        col -= ncols
                    z_new[row_c, col_c] += weight[j, i] * z[row, col]
    return z_new


def similarity_weight(sigma_c):
    lw = int(sigma_c * 3 + 0.5)
    weight = {}
    count = 0
    for x in range(-lw, lw + 1):
        weight[float(x)] = gauss(x, sigma_c)
        count += gauss(x, sigma_c)
    for key in weight.keys():
        weight[key] /= count
    return weight


def bilateral_filter(z, sigma_s, sigma_c):
    if isinstance(sigma_s, list):
        if len(sigma_s) == 2:
            sigma_x, sigma_y = sigma_s
        else:
            sigma_x = sigma_s
            sigma_y = sigma_x
    weight_space = gauss_weight(sigma_x, sigma_y)
    weight_color = similarity_weight(sigma_c)
    lw_y = (weight_space.shape[0] - 1) // 2
    lw_x = (weight_space.shape[1] - 1) // 2
    lw_c = (len(weight_color) - 1) // 2
    print(lw_y, lw_x, lw_c)
    z_new = np.zeros_like(z)
    nrows, ncols = z.shape
    for row_c in range(nrows):
        for col_c in range(ncols):
            zc = z[row_c, col_c]
            count = 0
            for j, drow in enumerate(range(-lw_y, lw_y + 1)):
                row = row_c + drow
                if row >= nrows:
                    row -= nrows
                for i, dcol in enumerate(range(-lw_x, lw_x + 1)):
                    col = col_c + dcol
                    if col >= ncols:
                        col -= ncols
                    dz = z[row, col] - zc
                    if -lw_c <= dz <= lw_c:
                        weight = weight_space[j, i] * weight_color[dz]
                        z_new[row_c, col_c] += weight * z[row, col]
                        count += weight
            z_new[row_c, col_c] /= count
    return z_new


if __name__ == "__main__":
    weight = similarity_weight(2)
    for key in weight:
        print(key, weight[key])
