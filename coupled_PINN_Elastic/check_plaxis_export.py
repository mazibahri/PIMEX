"""Sanity checks on a PLAXIS export before using it as a reference.

Run this first when you bring your own PLAXIS model. It checks that the tables can be read,
that the boundary fixities and the geostatic far field look right, that the wall traction
after the excavation phase is close to (1 - d1) * sigma0, and how the PLAXIS plate forces
relate to EA * strain and EI * curvature of the lining displacement.

    python check_plaxis_export.py --data data --model Elastic
"""
import argparse
import os

import numpy as np
from scipy.spatial import cKDTree

from plaxis_io import (read_soil_displacements, read_soil_stresses, read_plate, plate_unique)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data")
    p.add_argument("--model", default="Elastic", help="file name prefix of the PLAXIS tables")
    p.add_argument("--gamma", type=float, default=15.0)
    p.add_argument("--K0", type=float, default=0.6)
    p.add_argument("--d1", type=float, default=0.6)
    p.add_argument("--a", type=float, default=2.5)
    p.add_argument("--yc", type=float, default=-17.0)
    p.add_argument("--L", type=float, default=35.0)
    p.add_argument("--Hb", type=float, default=30.0)
    p.add_argument("--EA", type=float, default=14e6)
    p.add_argument("--EI", type=float, default=143e3)
    p.add_argument("--nu_plate", type=float, default=0.15)
    A = p.parse_args()
    D = lambda s: os.path.join(A.data, f"{A.model}_{s}.TXT")
    g, K0, a, yc = A.gamma, A.K0, A.a, A.yc

    dE = read_soil_displacements(D("Soil_displacements_TableTotalDisplacements_ExcPhase"))
    dL = read_soil_displacements(D("Soil_displacements_TableTotalDisplacements_LinPhase"))
    sE = read_soil_stresses(D("Soil_stresses_TableEffectiveStresses_ExcPhase"))
    pf = plate_unique(read_plate(D("Plate_forces_TablePlateForces_LinPhase")), yc=yc)
    pu = plate_unique(read_plate(D("Plate_displacements_TablePlateDisplacements_LinPhase")), yc=yc)

    print(f"nodes: {len(dE)} (exc), {len(dL)} (lin); stress points: {len(sE)}; plate nodes: {len(pf)}")
    print(f"domain: x {dE.x.min():.2f} .. {dE.x.max():.2f}, y {dE.y.min():.2f} .. {dE.y.max():.2f}")
    print(f"smallest node radius from tunnel centre: {np.hypot(dE.x, dE.y - yc).min():.4f} m (expect {a})")

    # boundary fixities and surface trough
    for name, d in (("excavation", dE), ("lining", dL)):
        left = np.abs(d.ux[d.x < 1e-9]).max()
        right = np.abs(d.ux[d.x > A.L - 1e-9]).max()
        bot = d.y < -A.Hb + 1e-9
        bottom = np.abs(np.r_[d.ux[bot], d.uy[bot]]).max()
        s = d[d.y > -1e-9].sort_values("x")
        trough = [round(float(np.interp(x, s.x, s.uy)) * 1e3, 2) for x in (0, 5, 10, 20, A.L)]
        print(f"{name}: max|ux| left {left:.1e}, right {right:.1e}, max|u| bottom {bottom:.1e}")
        print(f"   surface u_y at x = 0, 5, 10, 20, L: {trough} mm")

    # far field should still be geostatic after the excavation phase
    rs = np.hypot(sE.x, sE.y - yc)
    for lo, hi in ((12, 15), (20, 40)):
        m = (rs > lo) & (rs < hi)
        print(f"far field r in [{lo}, {hi}] m: syy/(g y) = {np.mean(sE.syy[m] / (g * sE.y[m])):.3f}, "
              f"sxx/(K0 g y) = {np.mean(sE.sxx[m] / (K0 * g * sE.y[m])):.3f}")

    # radial stress next to the wall after excavation, relative to sigma0 (expect 1 - d1)
    m = rs < a + 0.12
    th = np.arctan2(sE.y[m] - yc, sE.x[m]); c, s_ = np.cos(th), np.sin(th)
    srr = sE.sxx[m] * c**2 + sE.syy[m] * s_**2 + 2 * sE.sxy[m] * s_ * c
    srr0 = K0 * g * sE.y[m] * c**2 + g * sE.y[m] * s_**2
    print(f"wall: srr/srr0 = {np.mean(srr / srr0):.3f} +- {np.std(srr / srr0):.3f} "
          f"(expect about {1 - A.d1:.2f}, {m.sum()} stress points)")

    # lining: incremental displacement since installation, fitted with a Fourier series
    dist, idx = cKDTree(np.c_[dE.x, dE.y]).query(np.c_[pu.x, pu.y])
    print(f"plate-to-soil node match, max distance: {dist.max():.2e} m (0 means bonded, shared nodes)")
    dux = pu.ux.values - dE.ux.values[idx]
    duy = pu.uy.values - dE.uy.values[idx]
    th = pu.theta.values
    ur = dux * np.cos(th) + duy * np.sin(th)
    ut = -dux * np.sin(th) + duy * np.cos(th)
    phi = th - np.pi / 2                       # symmetric about the vertical axis
    n = np.arange(0, 9)
    Ar = np.linalg.lstsq(np.cos(np.outer(phi, n)), ur, rcond=None)[0]
    Bt = np.linalg.lstsq(np.sin(np.outer(phi, n[1:])), ut, rcond=None)[0]

    pp = pf.theta.values - np.pi / 2
    urp = np.cos(np.outer(pp, n)) @ Ar
    ur2 = -(np.cos(np.outer(pp, n)) * n**2) @ Ar
    utp = (np.cos(np.outer(pp, n[1:])) * n[1:]) @ Bt
    eps, kap = (utp + urp) / a, (utp - ur2) / a**2
    N, M = pf.N.values, pf.M.values
    print(f"N/(EA eps) median: {np.median(N / (A.EA * eps)):.4f}   (1/(1-nu^2) = {1 / (1 - A.nu_plate**2):.4f})")
    big = np.abs(kap) > 0.3 * np.abs(kap).max()
    print(f"-M/(EI kappa) median: {np.median(-M[big] / (A.EI * kap[big])):.4f}   (PLAXIS M sign is opposite to ours)")
    for name, t in (("crown", np.pi / 2), ("springline", 0.0), ("invert", -np.pi / 2)):
        print(f"{name:10s} N = {np.interp(t, pf.theta, pf.N):8.1f} kN/m   M = {np.interp(t, pf.theta, pf.M):7.1f} kNm/m")


if __name__ == "__main__":
    main()
