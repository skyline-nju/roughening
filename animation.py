""" Make an animation to show the growthing of band.

    Meanwhile, output the time serials of order parameters and width
    of interface.
"""

import os
import numpy as np
import matplotlib
import platform
matplotlib.use("Agg")


def make_movie(frames, file, out_data=False):
    def update_frame(frame):
        t, vxm, vym, num = frame
        rho = num.astype(float)
        rho_s = gaussian_filter(rho, sigma=[5, 1])
        xh1, rho_h = half_peak.find_interface(rho, sigma=[5, 1])
        xh2, rho_h = half_peak.find_interface(rho, sigma=[10, 1])
        xh3, rho_h = half_peak.find_interface(rho, sigma=[15, 1])
        xh1 = half_rho.untangle(xh1, Lx)
        xh2 = half_rho.untangle(xh2, Lx)
        xh3 = half_rho.untangle(xh3, Lx)
        w1 = np.var(xh1)
        w2 = np.var(xh2)
        w3 = np.var(xh3)
        dx = np.round(100 - np.mean(xh1)).astype(int)
        xh1 += dx
        xh2 += dx
        xh3 += dx
        rho_s = np.roll(rho_s, dx, axis=1)
        phi = np.sqrt(vxm**2 + vym**2)
        im.set_data(rho_s.T)
        line1.set_data(yh, xh1)
        # line2.set_data(yh, xh2)
        line3.set_data(yh, xh3)
        title.set_text(title_template % (eta, eps, Lx, Ly, t, phi, w1, w3))
        writer.grab_frame()
        print("t=", t)
        if out_data:
            f.write("%d\t%f\t%f\t%f\t%f\n" % (t, phi, w1, w2, w3))

    import half_peak
    import half_rho
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from scipy.ndimage import gaussian_filter
    if platform.system() is "Windows":
        plt.rcParams['animation.ffmpeg_path'] = r"D:\ffmpeg\bin\ffmpeg"
    else:
        plt.rcParams['animation.ffmpeg_path'] = "/home-yw/users/nsyw449_YK" \
            + "/dy/Program/ffmpeg-3.3-64bit-static/ffmpeg"
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
    # line2, = plt.plot([], [], lw=1.5, c="g")
    line3, = plt.plot([], [], lw=1.5, c="k")
    title = plt.title("")
    title_template = r"$\eta=%g, \epsilon=%g, L_x=%d, L_y=%d, t=%d," \
        + r"\phi=%.4f, w^2(\sigma_y=5)=%.4f, w^2(\sigma_y=15)=%.4f$"
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if out_data:
        f = open(file.replace(".mp4", ".dat"), "w")
    with writer.saving(fig, file, dpi=100):
        for frame in frames:
            update_frame(frame)
    if out_data:
        f.close()


if __name__ == "__main__":
    import sys
    import load_snap
    if platform.system() is "Windows":
        os.chdir("D:\\tmp")
        eta = 0.35
        eps = 0
        Lx = 180
        Ly = 25600
        seed = 4232
        interval = 1
    else:
        os.chdir("coarse")
        eta = float(sys.argv[1])
        eps = float(sys.argv[2])
        Lx = int(sys.argv[3])
        Ly = int(sys.argv[4])
        seed = int(sys.argv[5])
        if len(sys.argv) == 7:
            interval = int(sys.argv[6])
        else:
            interval = 1

    yh = np.arange(Ly) + 0.5
    file = "cB_%g_%g_%d_%d_%d_%d_%d_1.06_%d.bin" % (eta, eps, Lx, Ly, Lx, Ly,
                                                    Lx * Ly, seed)
    snap = load_snap.CoarseGrainSnap(file)
    frames = snap.gene_frames(interval=interval)
    make_movie(frames, file.replace("bin", "mp4"), out_data=True)
