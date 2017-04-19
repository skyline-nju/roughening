""" Calculate the interface with B-Tree.

    When using half_peak to detect the interface of band, it's hard to decide
    whether shift to nearest peak. A promising solution is to traverse all
    possible paths stored in a binary tree, and select the best one.

    However, it's a problem how to choose the best one.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from half_peak import detect_left_peak, detect_right_peak, get_idx_nearest, \
        cal_dis, find_first_row


class Node:
    def __init__(
            self, idx_h, rho_h, start_row, length=0, lchild=None, rchild=None):
        self.rho_h = [rho_h]
        self.col_list = [idx_h]
        self.start_row = start_row
        self.length = length
        self.lchild = lchild
        self.rchild = rchild

    def has_no_child(self):
        return self.lchild is None and self.rchild is None

    def has_one_child(self):
        return (self.lchild is None and self.rchild is not None) or \
                (self.lchild is not None and self.rchild is None)

    def enlongate(self, rho_x, debug=0):
        dh = 0.5  # threshold value for relative height between vally and peak
        idx_h_pre = self.col_list[-1]
        v_min = 0.5 * self.rho_h[-1]  # min height of valley
        idx_p1 = detect_left_peak(rho_x, idx_h_pre, dh)
        rho_h1 = rho_x[idx_p1] * 0.5
        idx_h1 = get_idx_nearest(rho_x, idx_p1, rho_h1)
        dx1 = cal_dis(idx_h1, idx_h_pre, rho_x.size)
        idx_p2 = detect_right_peak(rho_x, idx_h_pre, v_min, dh)
        if idx_p2 is None:
            self.rho_h.append(rho_h1)
            self.col_list.append(idx_h1)
            self.length += dx1**2
        else:
            rho_h2 = rho_x[idx_p2] * 0.5
            idx_h2 = get_idx_nearest(rho_x, idx_p2, rho_h2)
            dx2 = cal_dis(idx_h2, idx_h_pre, rho_x.size)
            start_row = self.start_row + len(self.col_list)
            self.lchild = Node(idx_h1, rho_h1, start_row, dx1**2, None, None)
            self.rchild = Node(idx_h2, rho_h2, start_row, dx2**2, None, None)

    def merge(self, is_left=True):
        if is_left:
            self.col_list.extend(self.lchild.col_list)
            self.rho_h.extend(self.lchild.rho_h)
            self.length += self.lchild.length
        else:
            self.col_list.extend(self.rchild.col_list)
            self.rho_h.extend(self.rchild.rho_h)
            self.length += self.rchild.length
        self.lchild = None
        self.rchild = None


class Tree:
    def __init__(self, rho, sigma):
        self.rho = gaussian_filter(rho, sigma=sigma, mode="wrap")
        start_row, idx_peak = find_first_row(self.rho)
        if idx_peak is not None:
            start_idx_h = get_idx_nearest(self.rho[start_row], idx_peak,
                                          self.rho[start_row, idx_peak] * 0.5)
            rho_h = self.rho[start_row, idx_peak] * 0.5
            self.root = Node(start_idx_h, rho_h, start_row)
            self.start_row = start_row

    def grow(self, node, rho_x):
        if node is None:
            return
        self.grow(node.rchild, rho_x)
        self.grow(node.lchild, rho_x)
        if node.has_no_child():
            node.enlongate(rho_x)
        if node.has_one_child():
            print("node has one child")

    def merge_close_loop(self, node):
        if node is None:
            return
        self.merge_close_loop(node.lchild)
        self.merge_close_loop(node.rchild)
        if node.lchild is not None and node.lchild.has_no_child() and \
                node.rchild is not None and node.rchild.has_no_child():
            if node.lchild.col_list[-1] == node.rchild.col_list[-1]:
                if node.lchild.length <= node.rchild.length:
                    node.merge(is_left=True)
                else:
                    node.merge(is_left=False)

    def filter(self, node):
        if node is None:
            return
        self.filter(node.lchild)
        self.filter(node.rchild)
        if node.lchild is not None and node.lchild.has_no_child() and \
                node.rchild is not None and node.rchild.has_no_child():
                col0 = self.root.col_list[0]
                col1 = node.lchild.col_list[-1]
                col2 = node.rchild.col_list[-1]
                ncols = self.rho.shape[1]
                dx1 = cal_dis(col1, col0, ncols)
                dx2 = cal_dis(col2, col0, ncols)
                if dx1 < dx2:
                    node.merge(is_left=True)
                else:
                    node.merge(is_left=False)

    def get_depth(self, node):
        if node is None:
            return 0
        max_left = self.get_depth(node.lchild)
        max_right = self.get_depth(node.rchild)
        if node.has_one_child():
            print("node has one child")
        return max(max_left, max_right) + 1

    def get_node_count(self, node):
        if node is None:
            return 0
        n_left = self.get_node_count(node.lchild)
        n_right = self.get_node_count(node.rchild)
        return n_left + n_right + 1

    def show(self, node=None):
        def traverse(node, ax):
            if node is None:
                return
            x = node.col_list
            y = np.arange(len(x)) + node.start_row
            y[y > nrows] -= nrows
            ax.plot(y, x, "o")
            traverse(node.lchild, ax)
            traverse(node.rchild, ax)
        if node is None:
            node = self.root
        nrows = self.rho.shape[0]
        ax = plt.subplot()
        traverse(self.root, ax)
        rho = self.rho.copy()
        rho[rho > 4] = 4
        plt.imshow(rho.T, interpolation="none", origin="lower")
        plt.show()
        plt.close()

    def eval(self):
        nrows = self.rho.shape[0]
        for row in range(self.start_row + 1, self.start_row + nrows):
            row = row % nrows
            self.grow(self.root, self.rho[row])
            self.merge_close_loop(self.root)
        self.show()
        if nrows > len(self.root.col_list):
            self.filter(self.root)
        self.show()


if __name__ == "__main__":
    import os
    import load_snap
    os.chdir(r"D:\tmp")
    Lx = 150
    Ly = 250
    snap = load_snap.RawSnap(r"so_%g_%g_%d_%d_%d_%d_%d.bin" %
                             (0.35, 0, Lx, Ly, Lx * Ly, 2000, 1234))
    debug = 1
    t_beg = 2409
    t_end = 2410
    for i, frame in enumerate(snap.gene_frames(t_beg, t_end)):
        x, y, theta = frame
        rho = load_snap.coarse_grain2(x, y, theta, Lx=Lx, Ly=Ly).astype(float)
        yh = np.linspace(0, Ly - 1, Ly)
        bt = Tree(rho, sigma=[5, 1])
        bt.eval()
