"""Inverse problems built on the forward model in pinn.py.

InverseE   - stage 1 with the soil Young's modulus E unknown. Data: a few displacement
             readings (surface levelling + tunnel convergence) from the excavation phase.
InverseD1  - stage 2 (coupled soil + lining) with the release factor d1 unknown and E known.
             Data: axial force N and bending moment M at a few lining gauges.

In both cases the physics terms are exactly those of pinn.Stage1 / pinn.Stage2; the only
changes are the trainable scalar and an extra data term in the loss. The scalar is
parametrised so it stays physical: E = E0 * exp(q), d1 = sigmoid(logit(d1_0) + z).

Synthetic "measurements" are taken from the PLAXIS export (nearest node / plate node).
All other PLAXIS values are withheld during training and only used for evaluation.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial import cKDTree

from pinn import (B01Config, GroundNet, Stage2, grad, ground_residuals, interior_points,
                  wall_points, polar_traction, sigma0)
from plaxis_io import read_soil_displacements, read_plate, plate_unique

# lining gauges: crown, shoulder, springline, knee, invert (theta from +x, CCW)
GAUGES_DEG = (90.0, 45.0, 0.0, -45.0, -90.0)


# ----------------------------------------------------------------------------- data
def sensor_data(disp_file, cfg: B01Config, noise=0.0, seed=0):
    """Displacement sensors from the excavation phase.

    16 surface settlement points (u_y at x = 0, 2, ..., 30 m) plus crown u_y, invert u_y
    and springline u_x. Noise is multiplicative: u_obs = u * (1 + noise * N(0, 1)).
    """
    d = read_soil_displacements(disp_file)
    tree = cKDTree(np.c_[d.x, d.y])
    pts, comp = [], []                      # comp: 0 -> u_x, 1 -> u_y
    for xs in np.arange(0, 31, 2.0):
        pts.append((xs, 0.0)); comp.append(1)
    pts += [(0.0, cfg.yc + cfg.a), (0.0, cfg.yc - cfg.a), (cfg.a, cfg.yc)]
    comp += [1, 1, 0]
    dist, idx = tree.query(np.array(pts))
    xy = np.c_[d.x.values[idx], d.y.values[idx]]
    val = np.where(np.array(comp) == 0, d.ux.values[idx], d.uy.values[idx])
    rng = np.random.default_rng(1000 + seed)
    obs = val * (1 + noise * rng.standard_normal(len(val)))
    return xy, np.array(comp), val, obs, float(dist.max())


def gauge_data(plate_forces_file, noise=0.0, seed=0, gauges_deg=GAUGES_DEG):
    """Lining N and M at the plate nodes closest to the gauge angles.

    M is returned in the sign convention of this code (M_code = -M_PLAXIS); N is the same.
    Noise is additive and scaled by the largest gauge value, so gauges near a zero crossing
    of M are not artificially noise-free.
    """
    pf = plate_unique(read_plate(plate_forces_file))
    th_all = pf.theta.values
    idx = [int(np.argmin(np.abs(th_all - math.radians(g)))) for g in gauges_deg]
    th = th_all[idx]
    N = pf.N.values[idx].astype(float)
    M = -pf.M.values[idx].astype(float)
    rng = np.random.default_rng(2000 + seed)
    zN, zM = rng.standard_normal(len(idx)), rng.standard_normal(len(idx))
    return dict(theta=th, N=N, M=M,
                N_obs=N + noise * np.abs(N).max() * zN,
                M_obs=M + noise * np.abs(M).max() * zM,
                snap_deg=float(np.max(np.abs(np.rad2deg(th) - np.array(gauges_deg)))))


# ----------------------------------------------------------------------------- E from displacements
class InverseE:
    """Stage 1 with E trainable.

    Only the constitutive residual depends on E (stresses follow from the loads alone and
    displacements scale with 1/E), so E is identifiable from displacements once gamma, K0,
    d1 and nu are fixed.
    """

    def __init__(self, cfg: B01Config, xy, comp, obs, E0, w_data=10.0, device="cpu", dtype=torch.float64):
        self.cfg, self.device, self.dtype = cfg, device, dtype
        torch.manual_seed(cfg.seed)
        self.rng = np.random.default_rng(cfg.seed)
        self.E0 = E0
        self.q = nn.Parameter(torch.zeros((), device=device, dtype=dtype))
        self.s_ref = cfg.d1 * cfg.sig_ref
        G0 = E0 / (2 * (1 + cfg.nu))
        self.u_ref = self.s_ref * cfg.a / (2 * G0)      # from the initial guess, kept fixed
        self.net = GroundNet(cfg, self.u_ref, self.s_ref).to(device, dtype)
        self.x, self.y = interior_points(cfg, self.rng, device, dtype)
        xw, yw, thw = wall_points(cfg, cfg.n_wall, device, dtype)
        self.xw, self.yw, self.thw = xw.requires_grad_(True), yw.requires_grad_(True), thw
        t = lambda v: torch.as_tensor(np.asarray(v, dtype=float), device=device, dtype=dtype)
        self.xd, self.yd = t(xy[:, 0]), t(xy[:, 1])
        self.comp = torch.as_tensor(comp, device=device)
        self.obs = t(obs)
        self.u_scale = float(np.abs(obs).max())
        self.w_data = w_data
        self.unknown = self.q

    @property
    def E(self):
        return self.E0 * torch.exp(self.q)

    def losses(self):
        c = self.cfg
        E = self.E
        G = E / (2 * (1 + c.nu))
        lam = E * c.nu / ((1 + c.nu) * (1 - 2 * c.nu))
        ux, uy, sxx, syy, sxy = self.net(self.x, self.y)
        exx, eyy = grad(ux, self.x), grad(uy, self.y)
        gxy = grad(ux, self.y) + grad(uy, self.x)
        r_c = torch.stack([sxx - ((lam + 2 * G) * exx + lam * eyy),
                           syy - (lam * exx + (lam + 2 * G) * eyy),
                           sxy - G * gxy], -1)
        r_eq = torch.stack([grad(sxx, self.x) + grad(sxy, self.y),
                            grad(sxy, self.x) + grad(syy, self.y)], -1)
        L_eq = ((r_eq / (self.s_ref / c.eq_len)) ** 2).mean()
        L_c = ((r_c / self.s_ref) ** 2).mean()

        _, _, wxx, wyy, wxy = self.net(self.xw, self.yw)
        tr, tt = polar_traction(wxx, wyy, wxy, self.thw)
        s0xx, s0yy = sigma0(c, self.yw)
        tr0, tt0 = polar_traction(s0xx, s0yy, torch.zeros_like(s0xx), self.thw)
        L_w = (((tr + c.d1 * tr0) / self.s_ref) ** 2 + ((tt + c.d1 * tt0) / self.s_ref) ** 2).mean()

        dux, duy, _, _, _ = self.net(self.xd, self.yd)
        pred = torch.where(self.comp == 0, dux, duy)
        L_d = (((pred - self.obs) / self.u_scale) ** 2).mean()

        tot = c.w_eq * L_eq + c.w_c * L_c + c.w_wall * L_w + self.w_data * L_d
        return tot, {"eq": L_eq.item(), "c": L_c.item(), "wall": L_w.item(),
                     "data": L_d.item(), "E": E.item()}

    def parameters(self):
        return list(self.net.parameters()) + [self.q]


# ----------------------------------------------------------------------------- d1 from lining forces
class InverseD1(Stage2):
    """Stage 2 with the release d1 trainable (E known) and a lining-gauge data term.

    d1 only enters the lining load p = ((1 - d1) sigma0 + ds2) . e_r; the soil residuals are
    unchanged. Reference scales (s_ref, u_ref, N_ref, M_ref) come from d1_0 and stay fixed.

        L = w_eq L_eq + w_c L_c + w_beam L_beam + w_cpl L_cpl + w_d L_data
        L_data = mean((N - N_obs) / s_N)^2 + mean((M - M_obs) / s_M)^2
        w_d    = w_data0 * (0.01 / max(noise, 0.01))^2
    """

    def __init__(self, cfg: B01Config, gd: dict, d1_0: float, noise: float, w_data0=0.1,
                 use=("N", "M"), device="cpu", dtype=torch.float64):
        cfg.d1 = d1_0
        super().__init__(cfg, device=device, dtype=dtype)
        self.d1_0 = d1_0
        self.logit0 = math.log(d1_0 / (1 - d1_0))
        self.z = nn.Parameter(torch.zeros((), device=device, dtype=dtype))
        t = lambda v: torch.as_tensor(np.asarray(v, dtype=float), device=device, dtype=dtype)
        self.thd = t(gd["theta"]).clone().requires_grad_(True)
        self.N_obs, self.M_obs = t(gd["N_obs"]), t(gd["M_obs"])
        self.sN = float(np.abs(gd["N_obs"]).max())
        self.sM = float(np.abs(gd["M_obs"]).max())
        self.useN, self.useM = ("N" in use), ("M" in use)
        self.w_data = w_data0 * (0.01 / max(noise, 0.01)) ** 2
        self.unknown = self.z

    @property
    def d1(self):
        return torch.sigmoid(self.logit0 + self.z)

    def losses(self):
        c = self.cfg
        d1 = self.d1
        r_eq, r_c = ground_residuals(self.net, self.x, self.y)
        L_eq = ((r_eq / (self.s_ref / c.eq_len)) ** 2).mean()
        L_c = ((r_c / self.s_ref) ** 2).mean()

        ux, uy, sxx, syy, sxy = self.net(self.xw, self.yw)
        th = self.thw
        s0xx, s0yy = sigma0(c, self.yw)
        tr, tt = polar_traction(sxx + (1 - d1) * s0xx, syy + (1 - d1) * s0yy, sxy, th)
        tr = tr - c.w_plate * torch.sin(th)
        tt = tt - c.w_plate * torch.cos(th)
        b = self.beam_fields(th)
        r1 = b["N1"] + b["M1"] / c.a + c.a * tt
        r2 = b["N"] - b["M2"] / c.a - c.a * tr
        r4 = b["M"] - c.EI_eff * b["kappa"]
        L_b = ((r1 / self.N_ref) ** 2 + (r2 / self.N_ref) ** 2 + (r4 / self.M_ref) ** 2).mean()

        cth, sth = torch.cos(th), torch.sin(th)
        ur_s = ux * cth + uy * sth
        ut_s = -ux * sth + uy * cth
        L_cpl = (((ur_s - b["ur"]) / self.u_ref) ** 2 + ((ut_s - b["ut"]) / self.u_ref) ** 2).mean()

        bd = self.beam_fields(self.thd)
        L_d = torch.zeros((), device=d1.device, dtype=d1.dtype)
        if self.useN:
            L_d = L_d + (((bd["N"] - self.N_obs) / self.sN) ** 2).mean()
        if self.useM:
            L_d = L_d + (((bd["M"] - self.M_obs) / self.sM) ** 2).mean()

        tot = c.w_eq * L_eq + c.w_c * L_c + c.w_beam * L_b + c.w_cpl * L_cpl + self.w_data * L_d
        return tot, {"eq": L_eq.item(), "c": L_c.item(), "beam": L_b.item(), "cpl": L_cpl.item(),
                     "data": L_d.item(), "d1": d1.item()}

    def parameters(self):
        return list(self.net.parameters()) + list(self.beam.parameters()) + [self.z]


def train_inverse(m, n_adam, lr, lr_u, n_lbfgs, log=print, ckpt=None):
    """Adam with its own learning rate for the scalar unknown, then joint L-BFGS."""
    from pinn import train
    nets = [p for p in m.parameters() if p is not m.unknown]
    opt = torch.optim.Adam([{"params": nets, "lr": lr}, {"params": [m.unknown], "lr": lr_u}])
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=(1e-1) ** (1 / max(n_adam, 1)))
    hist = []
    for it in range(n_adam):
        opt.zero_grad()
        tot, parts = m.losses()
        tot.backward()
        opt.step(); sched.step()
        if it % 250 == 0 or it == n_adam - 1:
            hist.append(("adam", it, tot.item(), parts))
            log(f"adam {it:6d} {tot.item():.3e} {parts}")
    if n_lbfgs > 0:
        hist += [(h[0], n_adam + h[1], h[2], h[3])
                 for h in train(m, n_adam=0, n_lbfgs=n_lbfgs, log_every=250, log=log, ckpt=ckpt)]
    return hist
