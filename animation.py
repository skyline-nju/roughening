""" Make an animation to show the roughening of band.

    Meanwhile, output the time serials of order parameters and width
    of interface.

"""

import os
import numpy as np
import matplotlib
import platform
import sys
import load_snap
import half_peak
import half_rho
from scipy.ndimage import gaussian_filter
matplotlib.use("Agg")
if platform.system() is "Windows":
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    plt.rcParams['animation.ffmpeg_path'] = r"D:\ffmpeg\bin\ffmpeg"
else:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    dest = "/ffmpeg-3.3-64bit-static/ffmpeg"
    path1 = "/home-gk/users/nscc1185/Applications"
    path2 = "/home-yw/users/nsyw449_YK/dy/Program"
    if os.path.exists(path1):
        plt.rcParams['animation.ffmpeg_path'] = path1 + dest
    elif os.path.exists(path2):
        plt.rcParams['animation.ffmpeg_path'] = path2 + dest
    else:
        print("Error, cannot find ffmpeg")
        sys.exit()


def get_para(file):
    if isinstance(file, list):
        eta = []
        eps = []
        Lx = []
        Ly = []
        N = []
        for f in file:
            s = f.replace(".bin", "").split("_")
            eta.append(float(s[1]))
            eps.append(float(s[2]))
            Lx.append(int(s[3]))
            Ly.append(int(s[4]))
            N.append(int(s[7]))
    else:
        s = file.replace(".bin", "").split("_")
        eta = float(s[1])
        eps = float(s[2])
        Lx = int(s[3])
        Ly = int(s[4])
        # N = int(s[7])
        N = int(s[5])
        print("eta = %g, eps = %g, Lx = %d, Ly = %d, N = %d" %
              (eta, eps, Lx, Ly, N))
    return eta, eps, Lx, Ly, N


def make_movie_single(file, t_beg=0, t_end=None, interval=1, format=1):
    """ Make movie from one single file.

        Each frame contains the contour plot of density and curves of
        interfaces estimated by diffrent sigma_y.

        Parameters:
        --------
        file: str
            The filename of input data.
        t_beg, t_end, interval: int
            First, last and interval of frames.
        format: int
            1: each frame contains t, vxm, vym, num,
            2: each frame only contains t, phi, num.

    """

    def update_frame(frame, format=1):
        if format == 1:
            t, vxm, vym, num = frame
            phi = np.sqrt(vxm**2 + vym**2)
        elif format == 2:
            t, phi, num = frame
        rho = num.astype(float)
        rho_s = gaussian_filter(rho, sigma=[5, 1])
        xh1, rho_h = half_peak.find_interface(rho, sigma=[5, 1])
        xh2, rho_h = half_peak.find_interface(rho, sigma=[10, 1])
        xh3, rho_h = half_peak.find_interface(rho, sigma=[15, 1])
        xh4, rho_h = half_peak.find_interface(rho, sigma=[20, 1])
        xh1 = half_rho.untangle(xh1, Lx)
        xh2 = half_rho.untangle(xh2, Lx)
        xh3 = half_rho.untangle(xh3, Lx)
        xh4 = half_rho.untangle(xh4, Lx)
        w1 = np.var(xh1)
        w2 = np.var(xh2)
        w3 = np.var(xh3)
        w4 = np.var(xh4)
        dx = np.round(100 - np.mean(xh1)).astype(int)
        xh1 += dx
        xh2 += dx
        xh3 += dx
        rho_s = np.roll(rho_s, dx, axis=1)
        im.set_data(rho_s.T)
        line1.set_data(yh, xh1)
        line3.set_data(yh, xh3)
        title.set_text(title_template % (eta, eps, Lx, Ly, N, t, phi, w1, w3))
        writer.grab_frame()
        print("t=", t)
        f.write("%d\t%f\t%f\t%f\t%f\t%f\n" % (t, phi, w1, w2, w3, w4))

    eta, eps, Lx, Ly, N = get_para(file)
    yh = np.arange(Ly) + 0.5
    if format == 1:
        snap = load_snap.CoarseGrainSnap(file)
    elif format == 2:
        snap = load_snap.NewCoarseGrainSnap(file)
    file = file.replace("../c", "w")
    frames = snap.gene_frames(t_beg, t_end, interval=interval)
    FFMpegWriter = animation.writers["ffmpeg"]
    writer = FFMpegWriter(fps=4, metadata=dict(artist='Matplotlib'))
    fig = plt.figure(figsize=(16, 3.6))
    im = plt.imshow(
        np.zeros((Ly, Lx)),
        animated=True,
        interpolation=None,
        extent=[0, Ly, 0, Lx],
        origin="lower",
        vmin=0,
        vmax=5,
        aspect="auto")
    line1, = plt.plot([], [], lw=1.5, c="r")
    line3, = plt.plot([], [], lw=1.5, c="k")
    title = plt.title("")
    title_template = r"$\eta=%g, \epsilon=%g, L_x=%d, L_y=%d, N=%d, t=%d," \
        + r"\phi=%.4f, w^2(\sigma_y=5)=%.4f, w^2(\sigma_y=15)=%.4f$"
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    f = open(file.replace(".bin", ".dat"), "w")
    with writer.saving(fig, file.replace(".bin", ".mp4"), dpi=100):
        for frame in frames:
            update_frame(frame, format=format)
    f.close()


def make_movie_mult(files, t_beg=0, t_end=None):
    """ Make movie from multiple file.

        Each frame contains multiplt interfaces generated from each input
        file.
    """

    def update_frame():
        for i in range(nfile):
            t, vxm, vym, num = next(frames[i])
            xh, rho_h = half_peak.find_interface(
                num.astype(float), sigma=[20, 1])
            xh = half_rho.untangle(xh, Lx[i])
            w = np.var(xh)
            dx = 100 - np.mean(xh)
            xh += dx
            phi = np.sqrt(vxm**2 + vym**2)
            line[i].set_data(yh[i], xh)
            line[i].set_label(r"$L_y=%d, w^2=%.4f, \phi=%.4f$" %
                              (Lx[i], w, phi))
        title.set_text(r"$\eta=%g, \epsilon=%g, L_x=%d, t=%d$" %
                       (eta[0], eps[0], Ly[0], t))
        plt.legend()
        writer.grab_frame()
        print("t=", t)

    nfile = len(files)
    eta, eps, Lx, Ly, N = get_para(files)
    yh = [np.arange(ly) + 0.5 for ly in Ly]
    snap = [load_snap.CoarseGrainSnap(file) for file in files]
    i_end = min(i.get_num_frames() for i in snap)
    if t_end is not None:
        t_end = min(i_end, t_end)
    else:
        t_end = i_end
    frames = [i.gene_frames(t_beg, t_end) for i in snap]
    FFMpegWriter = animation.writers["ffmpeg"]
    writer = FFMpegWriter(fps=4, metadata=dict(artist='Matplotlib'))
    fig = plt.figure(figsize=(16, 3))
    plt.xlim(0, Ly[0])
    plt.ylim(0, 400)
    line = []
    c_list = plt.cm.viridis(np.linspace(0, 1, nfile))
    for i in range(nfile):
        l, = plt.plot([], [], lw=1.5, c=c_list[i])
        line.append(l)
    title = plt.title("")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    with writer.saving(fig, "%g_%g_%d.mp4" % (eta[0], eps[0], Ly[0]), dpi=100):
        for j in range(t_end - t_beg):
            update_frame()


if __name__ == "__main__":
    if platform.system() is "Windows":
        os.chdir("D:\\tmp")
        eta = 0.35
        eps = 0.02
        Lx = 220
        Ly = 25600
        seed = 1234
        interval = 1
        N = Lx * Ly
    else:
        # os.chdir("coarse")
        if len(sys.argv) == 2:
            path = sys.argv[1]
            make_movie_single(path, format=2)
        elif len(sys.argv) > 2:
            files = [file.split("/")[1] for file in sys.argv[1:]]
            make_movie_mult(files)
