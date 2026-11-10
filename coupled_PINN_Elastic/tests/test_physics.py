"""Tests that need no PLAXIS data: residuals, sign conventions, hard BCs, beam equations.

    python -m pytest -q tests
"""
import math
import numpy as np
import torch

from pinn import (B01Config, ground_residuals, polar_traction, grad, Stage1, Stage2,
                      GroundNet)

torch.set_default_dtype(torch.float64)


class KirschFake(torch.nn.Module):
    """Excavation increments of an unsupported deep tunnel (Kirsch), full release.
    Far field (tension +): sxx0 = -K p, syy0 = -p. Standard Kirsch displacements in plane
    strain (outward positive); stresses = C:eps(u) via autograd."""

    def __init__(self, cfg, p=100.0, K=0.6):
        super().__init__()
        self.cfg, self.p, self.K = cfg, p, K

    def u(self, x, y):
        c = self.cfg
        dx, dy = x, y - c.yc
        r = torch.sqrt(dx**2 + dy**2); th = torch.atan2(dy, dx)
        a, G, nu, p, K = c.a, c.G, c.nu, self.p, self.K
        f = -(p * a**2 / (4 * G * r))
        ur = f * ((1 + K) - (1 - K) * (4 * (1 - nu) - a**2 / r**2) * torch.cos(2 * th))
        ut = f * (1 - K) * (2 * (1 - 2 * nu) + a**2 / r**2) * torch.sin(2 * th)
        return ur * torch.cos(th) - ut * torch.sin(th), ur * torch.sin(th) + ut * torch.cos(th)

    def forward(self, x, y):
        c = self.cfg
        ux, uy = self.u(x, y)
        exx, eyy = grad(ux, x), grad(uy, y)
        gxy = grad(ux, y) + grad(uy, x)
        sxx = (c.lam + 2 * c.G) * exx + c.lam * eyy
        syy = c.lam * exx + (c.lam + 2 * c.G) * eyy
        return ux, uy, sxx, syy, c.G * gxy


def test_kirsch_equilibrium_and_wall_traction():
    cfg = B01Config()
    fake = KirschFake(cfg)
    rng = np.random.default_rng(1)
    r = cfg.a * (1 + 4 * rng.random(500)); th = rng.uniform(-math.pi / 2, math.pi / 2, 500)
    x = torch.tensor(r * np.cos(th), requires_grad=True)
    y = torch.tensor(cfg.yc + r * np.sin(th), requires_grad=True)
    r_eq, r_c = ground_residuals(fake, x, y)
    assert r_eq.abs().max().item() < 1e-8 * fake.p
    assert r_c.abs().max().item() < 1e-10 * fake.p
    # traction-free total stress at the wall: dsigma . e_r = - sigma0 . e_r
    thw = torch.linspace(-math.pi / 2, math.pi / 2, 101)
    xw = (cfg.a * torch.cos(thw)).requires_grad_(True)
    yw = (cfg.yc + cfg.a * torch.sin(thw)).requires_grad_(True)
    _, _, sxx, syy, sxy = fake(xw, yw)
    tr, tt = polar_traction(sxx, syy, sxy, thw)
    tr0, tt0 = polar_traction(-fake.K * fake.p * torch.ones_like(thw), -fake.p * torch.ones_like(thw),
                              torch.zeros_like(thw), thw)
    assert (tr + tr0).abs().max().item() < 1e-9 * fake.p
    assert (tt + tt0).abs().max().item() < 1e-9 * fake.p


def test_hard_bcs_and_symmetry():
    cfg = B01Config(width=16, depth=2)
    net = GroundNet(cfg, 0.05, 150.0)
    with torch.no_grad():
        for p in net.parameters():
            p.normal_(0, 0.5)
    yy = torch.linspace(-30, 0, 50)
    xx = torch.linspace(0, 35, 50)
    ux, uy, sxx, syy, sxy = net(torch.zeros(50), yy); assert ux.abs().max() == 0 and sxy.abs().max() == 0
    ux, uy, sxx, syy, sxy = net(35 * torch.ones(50), yy); assert ux.abs().max() < 1e-14 and sxy.abs().max() < 1e-12
    ux, uy, sxx, syy, sxy = net(xx, -30 * torch.ones(50)); assert ux.abs().max() < 1e-14 and uy.abs().max() < 1e-14
    ux, uy, sxx, syy, sxy = net(xx, torch.zeros(50)); assert syy.abs().max() == 0 and sxy.abs().max() == 0
    # exact evenness of features: f(-x) = f(x) for the MLP part
    f1 = net.features(xx, yy); f2 = net.features(-xx, yy)
    assert (f1 - f2).abs().max() < 1e-12


def test_ring_beam_equations():
    """Thin ring under p_r = q cos(2 theta): exact N = -q a cos2t/3, M = q a^2 cos2t/3
    must satisfy the strong form used in Stage2 (r1, r2)."""
    a, q = 2.5, 30.0
    th = torch.linspace(-math.pi / 2, math.pi / 2, 201, requires_grad=True)
    N = -q * a * torch.cos(2 * th) / 3
    M = q * a**2 * torch.cos(2 * th) / 3
    N1, M1 = grad(N, th), grad(M, th); M2 = grad(M1, th)
    r1 = N1 + M1 / a + a * 0.0
    r2 = N - M2 / a - a * q * torch.cos(2 * th)
    assert r1.abs().max() < 1e-10 and r2.abs().max() < 1e-10


def test_stage_losses_finite():
    cfg = B01Config(n_uniform=200, n_ring=200, n_wall=32, width=16, depth=2)
    for S in (Stage1, Stage2):
        m = S(cfg)
        tot, parts = m.losses()
        tot.backward()
        assert math.isfinite(tot.item())


def test_beam_symmetry():
    s2 = Stage2(B01Config(n_uniform=50, n_ring=50, n_wall=16, width=8, depth=1))
    th = torch.tensor([math.pi / 2, -math.pi / 2], requires_grad=True)
    b = s2.beam_fields(th)
    assert b["ut"].abs().max() < 1e-14            # u_t = 0 at crown and invert
    assert b["M1"].abs().max() < 1e-10            # shear Q = -M'/a = 0 at crown and invert
    assert b["N1"].abs().max() < 1e-10


if __name__ == "__main__":
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            v(); print("PASS", k)
