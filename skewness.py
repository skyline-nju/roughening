import numpy as np
import sys
import os


def skew(x):
    xm = np.mean(x)
    dx = x - xm
    gamma_1 = np.mean(dx ** 3) / np.mean(dx ** 2) ** (3/2)
    return gamma_1


def kurt(x):
    xm = np.mean(x)
    dx = x - xm
    gamma_2 = np.mean(dx ** 4) / np.mean(dx ** 2) ** 2 - 3
    return gamma_2


if __name__ == "__main__":
    import load_snap
    import half_peak
    import half_rho
    from cal_width_old import isoline
    path, file = sys.argv[1].split("/")
    s = file.replace(".bin", "").split("_")
    Lx = int(s[3])
    Ly = int(s[4])
    os.chdir(path)
    outfile = file.replace(".bin", "_skew.dat")
    f = open(outfile, "w")
    snap = load_snap.CoarseGrainSnap(file)
    tot_frames = snap.get_num_frames()
    ts = np.zeros(tot_frames, int)
    h10 = np.zeros((tot_frames, Ly))
    h15 = np.zeros((tot_frames, Ly))

    frames = snap.gene_frames()
    for i, frame in enumerate(frames):
        t, vxm, vym, num = frame
        ts[i] = t
        line = "%d" % t
        rho = num.astype(float)
        xh1, rho_h = half_peak.find_interface(rho, sigma=[10, 1])
        xh1 = half_rho.untangle(xh1, Lx)
        h10[i] = xh1
        gamma1 = skew(xh1)
        gamma2 = kurt(xh1)
        line += "\t%f\t%f" % (gamma1, gamma2)
        xh1, rho_h = half_peak.find_interface(rho, sigma=[15, 1])
        xh1 = half_rho.untangle(xh1, Lx)
        h15[i] = xh1
        gamma1 = skew(xh1)
        gamma2 = kurt(xh1)
        line += "\t%f\t%f" % (gamma1, gamma2)

        try:
            xh2 = isoline(rho, Lx, Ly, Lx, Ly//20)
            gamma1 = skew(xh2)
            gamma2 = kurt(xh2)
            line += "\t%f\t%f\n" % (gamma1, gamma2)
        except:
            line += "\t\t\n"
        f.write(line)
    f.close()
    np.savez(outfile.replace("_skew.dat", ".npz"), ts=ts, h10=h10, h15=h15)
