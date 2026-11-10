"""Elastic ground + tunnel lining PINN (plane strain, half model).

Geometry follows the PLAXIS model: x in [0, L], y in [-Hb, 0] with y pointing up and the
ground surface at y = 0. Circular tunnel of radius a centred at (0, yc). Tension is positive.

Everything is solved as an increment on top of the K0 geostatic state
    sigma0 = diag(K0*g*y, g*y)      (sigma_zz0 = K0*g*y)
which is self-equilibrated, so the increments carry no body force.

Stage 1 - excavation, no lining, deconfinement d1:
    div(ds1) = 0,   ds1 = C : eps(du1)
    wall:  ds1 . e_r = -d1 * sigma0 . e_r

Stage 2 - lining installed, remaining release (1 - d1):
    soil:    div(ds2) = 0,   ds2 = C : eps(du2)
    bonded:  du2(wall) = lining displacement (u_r, u_t)
    load on the lining (per unit length, outward positive):
             p = ((1 - d1) * sigma0 + ds2) . e_r   + self-weight (0, -w_plate)
    curved Euler-Bernoulli beam at r = a, theta measured CCW from +x:
             N' + M'/a + a p_t = 0
             N  - M''/a - a p_r = 0
             N  = EA (u_t' + u_r) / a        imposed exactly via u_r = a N / EA - u_t'
             M  = EI (u_t' - u_r'') / a^2    penalised in the loss

Straight boundaries use hard constraints (PLAXIS default fixities):
    x = 0 and x = L : u_x = 0, s_xy = 0
    y = -Hb         : u_x = u_y = 0
    y = 0           : s_yy = s_xy = 0
Symmetry about x = 0 is exact because the network inputs are even in x.

Because the soil is linear elastic, stage 2 does not depend on stage 1 and the two stages
can be trained separately; totals are stage 1 + stage 2.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn


@dataclass
class B01Config:
    L: float = 35.0
    Hb: float = 30.0
    a: float = 2.5
    yc: float = -17.0
    E: float = 10_000.0          # kPa
    nu: float = 0.33
    gamma: float = 15.0          # kN/m3
    K0: float = 0.6
    d1: float = 0.6              # deconfinement before lining
    EA: float = 14e6             # kN/m
    EI: float = 143e3            # kN m2/m
    nu_plate: float = 0.15
    plate_plane_strain_factor: bool = True   # use EA/(1-nu^2), EI/(1-nu^2) as PLAXIS plates do in plane strain
    w_plate: float = 0.0         # lining self-weight (kN/m per m of lining), acts in -y from installation
    width: int = 64
    depth: int = 5
    n_uniform: int = 3000
    n_ring: int = 3000
    ring_rmax: float = 6.0       # in units of a
    n_wall: int = 256
    seed: int = 0
    w_eq: float = 1.0
    w_c: float = 1.0
    w_wall: float = 10.0
    w_beam: float = 1.0
    w_cpl: float = 10.0
    eq_len: float = 17.0         # length used to scale the equilibrium residual (m).
                                 # Use the tunnel depth, not a: scaling by a lets a small smooth
                                 # residual act as a spurious body force (check global balance).

    @property
    def G(self):
        return self.E / (2 * (1 + self.nu))

    @property
    def lam(self):
        return self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))

    @property
    def sig_ref(self):          # gamma * H at tunnel axis
        return self.gamma * (-self.yc)

    @property
    def EA_eff(self):
        return self.EA / (1 - self.nu_plate**2) if self.plate_plane_strain_factor else self.EA

    @property
    def EI_eff(self):
        return self.EI / (1 - self.nu_plate**2) if self.plate_plane_strain_factor else self.EI


# ----------------------------------------------------------------------------- networks
class MLP(nn.Module):
    def __init__(self, n_in, n_out, width, depth):
        super().__init__()
        layers, d = [], n_in
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.Tanh()]
            d = width
        layers.append(nn.Linear(d, n_out))
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, z):
        return self.net(z)


class GroundNet(nn.Module):
    """(x, y) -> (ux, uy, sxx, syy, sxy) in physical units (m, kPa); hard straight BCs."""

    def __init__(self, cfg: B01Config, u_ref: float, s_ref: float):
        super().__init__()
        self.cfg, self.u_ref, self.s_ref = cfg, u_ref, s_ref
        self.mlp = MLP(7, 5, cfg.width, cfg.depth)

    def features(self, x, y):
        c = self.cfg
        X2 = 2 * (x / c.L) ** 2 - 1
        Yn = 2 * (y + c.Hb) / c.Hb - 1
        dx, dy = x, y - c.yc
        r2 = dx**2 + dy**2
        r = torch.sqrt(r2)
        s = c.a / r
        sin1 = dy / r
        cos2 = (dx**2 - dy**2) / r2
        sin3 = 3 * sin1 - 4 * sin1**3
        cos4 = 2 * cos2**2 - 1
        return torch.stack([X2, Yn, s, s * sin1, s**2 * cos2, s**3 * sin3, s**4 * cos4], dim=-1)

    def forward(self, x, y):
        c = self.cfg
        o = self.mlp(self.features(x, y))
        fx = x / (x + c.a)                      # 0 at x=0 (odd factor)
        gL = (c.L - x) / c.L                     # 0 at x=L
        gB = (y + c.Hb) / c.Hb                   # 0 at bottom
        gS = (-y) / (-y + c.a)                   # 0 at surface
        ux = self.u_ref * fx * gL * gB * o[:, 0]
        uy = self.u_ref * gB * o[:, 1]
        sxx = self.s_ref * o[:, 2]
        syy = self.s_ref * gS * o[:, 3]
        sxy = self.s_ref * fx * gL * gS * o[:, 4]
        return ux, uy, sxx, syy, sxy


class BeamNet(nn.Module):
    """theta -> (u_t, N, M) with exact symmetry about the vertical axis.
    u_t = cos(theta) * G(sin theta);  N, M functions of sin(theta)."""

    def __init__(self, cfg: B01Config, u_ref, N_ref, M_ref, width=32, depth=4):
        super().__init__()
        self.cfg, self.u_ref, self.N_ref, self.M_ref = cfg, u_ref, N_ref, M_ref
        self.mlp = MLP(1, 3, width, depth)

    def forward(self, th):
        s = torch.sin(th).unsqueeze(-1)
        o = self.mlp(s)
        ut = self.u_ref * torch.cos(th) * o[:, 0]
        N = self.N_ref * o[:, 1]
        M = self.M_ref * o[:, 2]
        return ut, N, M


# ----------------------------------------------------------------------------- physics
def grad(f, x):
    return torch.autograd.grad(f, x, torch.ones_like(f), create_graph=True)[0]


def sigma0(cfg, y):
    return cfg.K0 * cfg.gamma * y, cfg.gamma * y      # sxx0, syy0 (szz0 = sxx0)


def ground_residuals(net: GroundNet, x, y):
    c = net.cfg
    ux, uy, sxx, syy, sxy = net(x, y)
    uxx, uxy = grad(ux, x), grad(ux, y)
    uyx, uyy = grad(uy, x), grad(uy, y)
    exx, eyy, gxy = uxx, uyy, uxy + uyx
    r_c = torch.stack([sxx - ((c.lam + 2 * c.G) * exx + c.lam * eyy),
                       syy - (c.lam * exx + (c.lam + 2 * c.G) * eyy),
                       sxy - c.G * gxy], -1)
    r_eq = torch.stack([grad(sxx, x) + grad(sxy, y),
                        grad(sxy, x) + grad(syy, y)], -1)
    return r_eq, r_c


def polar_traction(sxx, syy, sxy, th):
    c, s = torch.cos(th), torch.sin(th)
    tx = sxx * c + sxy * s
    ty = sxy * c + syy * s
    return tx * c + ty * s, -tx * s + ty * c          # t_r, t_theta  (sigma . e_r)


def wall_points(cfg, n, device, dtype):
    # midpoints on [-pi/2, pi/2] (right half of the ring)
    th = (torch.arange(n, device=device, dtype=dtype) + 0.5) / n * math.pi - math.pi / 2
    return cfg.a * torch.cos(th), cfg.yc + cfg.a * torch.sin(th), th


def interior_points(cfg, rng, device, dtype):
    pts = []
    n = 0
    while n < cfg.n_uniform:
        x = rng.uniform(0, cfg.L, 4 * cfg.n_uniform)
        y = rng.uniform(-cfg.Hb, 0, 4 * cfg.n_uniform)
        ok = (x**2 + (y - cfg.yc) ** 2) > cfg.a**2 * 1.0001
        pts.append(np.c_[x[ok], y[ok]])
        n += ok.sum()
    U = np.concatenate(pts)[: cfg.n_uniform]
    pts, n = [], 0
    while n < cfg.n_ring:
        r = cfg.a * np.exp(rng.uniform(0, math.log(cfg.ring_rmax), 4 * cfg.n_ring))
        th = rng.uniform(-math.pi / 2, math.pi / 2, 4 * cfg.n_ring)
        x, y = r * np.cos(th), cfg.yc + r * np.sin(th)
        ok = (y > -cfg.Hb) & (y < 0) & (r > cfg.a * 1.0001)
        pts.append(np.c_[x[ok], y[ok]])
        n += ok.sum()
    R = np.concatenate(pts)[: cfg.n_ring]
    P = np.concatenate([U, R])
    t = torch.tensor(P, device=device, dtype=dtype)
    return t[:, 0].clone().requires_grad_(True), t[:, 1].clone().requires_grad_(True)


# ----------------------------------------------------------------------------- stage 1
class Stage1:
    """Unlined excavation with deconfinement d1."""

    def __init__(self, cfg: B01Config, device="cpu", dtype=torch.float64):
        self.cfg, self.device, self.dtype = cfg, device, dtype
        torch.manual_seed(cfg.seed)
        self.rng = np.random.default_rng(cfg.seed)
        self.s_ref = cfg.d1 * cfg.sig_ref
        self.u_ref = self.s_ref * cfg.a / (2 * cfg.G)
        self.net = GroundNet(cfg, self.u_ref, self.s_ref).to(device, dtype)
        self.x, self.y = interior_points(cfg, self.rng, device, dtype)
        xw, yw, thw = wall_points(cfg, cfg.n_wall, device, dtype)
        self.xw, self.yw, self.thw = xw.requires_grad_(True), yw.requires_grad_(True), thw

    def wall_target(self):
        c = self.cfg
        s0xx, s0yy = sigma0(c, self.yw)
        tr0, tt0 = polar_traction(s0xx, s0yy, torch.zeros_like(s0xx), self.thw)
        return -c.d1 * tr0, -c.d1 * tt0

    def losses(self):
        c = self.cfg
        r_eq, r_c = ground_residuals(self.net, self.x, self.y)
        L_eq = ((r_eq / (self.s_ref / c.eq_len)) ** 2).mean()
        L_c = ((r_c / self.s_ref) ** 2).mean()
        _, _, sxx, syy, sxy = self.net(self.xw, self.yw)
        tr, tt = polar_traction(sxx, syy, sxy, self.thw)
        trt, ttt = self.wall_target()
        L_w = (((tr - trt) / self.s_ref) ** 2 + ((tt - ttt) / self.s_ref) ** 2).mean()
        tot = c.w_eq * L_eq + c.w_c * L_c + c.w_wall * L_w
        return tot, {"eq": L_eq.item(), "c": L_c.item(), "wall": L_w.item()}

    def parameters(self):
        return self.net.parameters()


# ----------------------------------------------------------------------------- stage 2
class Stage2:
    """Lining installed at d1; remaining (1-d1) release with bonded curved beam."""

    def __init__(self, cfg: B01Config, device="cpu", dtype=torch.float64):
        self.cfg, self.device, self.dtype = cfg, device, dtype
        torch.manual_seed(cfg.seed + 1)
        self.rng = np.random.default_rng(cfg.seed + 1)
        p2 = (1 - cfg.d1) * cfg.sig_ref
        self.s_ref = p2 * max(abs(1 - cfg.K0) / 2, 0.05)
        self.u_ref = self.s_ref * cfg.a / (2 * cfg.G)
        self.N_ref = p2 * cfg.a
        self.M_ref = 0.1 * p2 * cfg.a**2
        self.net = GroundNet(cfg, self.u_ref, self.s_ref).to(device, dtype)
        self.beam = BeamNet(cfg, self.u_ref, self.N_ref, self.M_ref).to(device, dtype)
        self.x, self.y = interior_points(cfg, self.rng, device, dtype)
        xw, yw, thw = wall_points(cfg, cfg.n_wall, device, dtype)
        self.xw, self.yw = xw.requires_grad_(True), yw.requires_grad_(True)
        self.thw = thw.clone().requires_grad_(True)

    def beam_fields(self, th):
        """u_r, u_t, N, M and derivatives along theta (th must require grad)."""
        c = self.cfg
        ut, N, M = self.beam(th)
        ut1 = grad(ut, th); ut2 = grad(ut1, th); ut3 = grad(ut2, th)
        N1 = grad(N, th); N2 = grad(N1, th)
        M1 = grad(M, th); M2 = grad(M1, th)
        ur = c.a * N / c.EA_eff - ut1                 # exact axial constitutive law
        ur2 = c.a * N2 / c.EA_eff - ut3
        kappa = (ut1 - ur2) / c.a**2
        return dict(ur=ur, ut=ut, N=N, M=M, N1=N1, M1=M1, M2=M2, kappa=kappa)

    def losses(self):
        c = self.cfg
        r_eq, r_c = ground_residuals(self.net, self.x, self.y)
        L_eq = ((r_eq / (self.s_ref / c.eq_len)) ** 2).mean()
        L_c = ((r_c / self.s_ref) ** 2).mean()
        # soil at wall
        ux, uy, sxx, syy, sxy = self.net(self.xw, self.yw)
        th = self.thw
        s0xx, s0yy = sigma0(c, self.yw)
        tr, tt = polar_traction(sxx + (1 - c.d1) * s0xx, syy + (1 - c.d1) * s0yy, sxy, th)
        b = self.beam_fields(th)
        # lining self-weight (0, -w) per unit length: p_r = -w sin(th), p_t = -w cos(th)
        tr = tr - c.w_plate * torch.sin(th)
        tt = tt - c.w_plate * torch.cos(th)
        # beam equilibrium and bending constitutive law
        r1 = b["N1"] + b["M1"] / c.a + c.a * tt
        r2 = b["N"] - b["M2"] / c.a - c.a * tr
        r4 = b["M"] - c.EI_eff * b["kappa"]
        L_b = ((r1 / self.N_ref) ** 2 + (r2 / self.N_ref) ** 2 + (r4 / self.M_ref) ** 2).mean()
        # bonded compatibility
        cth, sth = torch.cos(th), torch.sin(th)
        ur_s = ux * cth + uy * sth
        ut_s = -ux * sth + uy * cth
        L_cpl = (((ur_s - b["ur"]) / self.u_ref) ** 2 + ((ut_s - b["ut"]) / self.u_ref) ** 2).mean()
        tot = c.w_eq * L_eq + c.w_c * L_c + c.w_beam * L_b + c.w_cpl * L_cpl
        return tot, {"eq": L_eq.item(), "c": L_c.item(), "beam": L_b.item(), "cpl": L_cpl.item()}

    def parameters(self):
        return list(self.net.parameters()) + list(self.beam.parameters())


# ----------------------------------------------------------------------------- training
def train(model, n_adam=5000, lr=1e-3, n_lbfgs=2000, log_every=500, log=print, ckpt=None):
    params = list(model.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=(1e-2) ** (1 / max(n_adam, 1)))
    hist = []
    for it in range(n_adam):
        opt.zero_grad()
        tot, parts = model.losses()
        tot.backward()
        opt.step(); sched.step()
        if it % log_every == 0 or it == n_adam - 1:
            hist.append(("adam", it, tot.item(), parts)); log(f"adam {it:6d} {tot.item():.3e} {parts}")
    if n_lbfgs > 0:
        lb = torch.optim.LBFGS(params, lr=1.0, max_iter=50, history_size=50,
                               tolerance_grad=1e-14, tolerance_change=1e-16, line_search_fn="strong_wolfe")

        def closure():
            lb.zero_grad()
            t, _ = model.losses()
            t.backward()
            return t
        for k in range(n_lbfgs // 50):
            lb.step(closure)
            tot, parts = model.losses()
            hist.append(("lbfgs", (k + 1) * 50, tot.item(), parts))
            if ((k + 1) * 50) % log_every == 0:
                log(f"lbfgs {(k+1)*50:6d} {tot.item():.3e} {parts}")
            if ckpt:
                torch.save({k_: v.state_dict() for k_, v in _modules(model).items()}, ckpt)
    return hist


def _modules(model):
    d = {"net": model.net}
    if hasattr(model, "beam"):
        d["beam"] = model.beam
    return d


@torch.no_grad()
def predict_ground(net, x, y, batch=20000):
    out = []
    dev = next(net.parameters()).device
    x, y = np.array(x, dtype=np.float64), np.array(y, dtype=np.float64)
    for i in range(0, len(x), batch):
        xx = torch.as_tensor(x[i:i + batch], device=dev)
        yy = torch.as_tensor(y[i:i + batch], device=dev)
        out.append(torch.stack(net(xx, yy), -1).cpu().numpy())
    return np.concatenate(out)


def load_checkpoint(run_dir, stage, device="cpu"):
    """Rebuild a trained Stage1/Stage2 model from a run folder written by train.py."""
    import json, os
    with open(os.path.join(run_dir, "hist.json")) as f:
        h = json.load(f)
    cfg = B01Config(**h["cfg"])
    cfg.n_uniform = cfg.n_ring = 10      # collocation points are not needed for prediction
    cfg.n_wall = 8
    model = (Stage1 if stage == 1 else Stage2)(cfg, device=device)
    sd = torch.load(os.path.join(run_dir, "model.pt"), map_location=device)
    model.net.load_state_dict(sd["net"])
    if stage == 2:
        model.beam.load_state_dict(sd["beam"])
    return model, cfg, h
