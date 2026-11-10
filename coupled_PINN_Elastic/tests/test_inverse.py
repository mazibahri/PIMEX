"""Checks that the inverse models use exactly the forward physics.

The first two tests need no data. The others read the PLAXIS plate forces and are skipped
when data/ is empty.

    python -m pytest -q tests                    # uses ./data
    python -m pytest -q tests --data /path/to/plaxis/tables
"""
import math

import numpy as np
import torch

from pinn import B01Config, Stage1, Stage2
from inverse import InverseE, InverseD1, gauge_data, GAUGES_DEG

torch.set_default_dtype(torch.float64)

SMALL = dict(n_uniform=200, n_ring=200, n_wall=64)


def _fake_sensors():
    xy = np.array([[0.0, 0.0], [4.0, 0.0], [0.0, -14.5]])
    return xy, np.array([1, 1, 1]), np.array([-0.019, -0.015, -0.057])


def _fake_gauges():
    th = np.deg2rad(np.array(GAUGES_DEG))
    N = np.array([-141.0, -200.0, -254.0, -200.0, -150.0])
    M = np.array([-56.0, 2.0, 59.0, 1.0, -59.0])
    return dict(theta=th, N=N, M=M, N_obs=N, M_obs=M)


def test_inverse_E_physics_matches_stage1():
    """At q = 0 (E = E0) the physics terms must equal Stage1.losses exactly."""
    xy, comp, obs = _fake_sensors()
    for E0 in (5000.0, 20000.0):
        ref = Stage1(B01Config(seed=3, E=E0, **SMALL))
        inv = InverseE(B01Config(seed=3, **SMALL), xy, comp, obs, E0=E0)
        inv.net.load_state_dict(ref.net.state_dict())
        _, pr = ref.losses()
        _, pi = inv.losses()
        for k in ("eq", "c", "wall"):
            assert abs(pr[k] - pi[k]) <= 1e-12 * max(1.0, abs(pr[k])), (E0, k)
        assert abs(pi["E"] - E0) < 1e-9


def test_d1_physics_matches_stage2():
    """At z = 0 (d1 = d1_0) the physics terms must equal Stage2.losses exactly."""
    gd = _fake_gauges()
    for d1 in (0.4, 0.6, 0.8):
        ref = Stage2(B01Config(seed=3, w_plate=8.4, d1=d1, **SMALL))
        inv = InverseD1(B01Config(seed=3, w_plate=8.4, **SMALL), gd, d1_0=d1, noise=0.0)
        inv.net.load_state_dict(ref.net.state_dict())
        inv.beam.load_state_dict(ref.beam.state_dict())
        _, pr = ref.losses()
        _, pi = inv.losses()
        for k in ("eq", "c", "beam", "cpl"):
            assert abs(pr[k] - pi[k]) <= 1e-12 * max(1.0, abs(pr[k])), (d1, k)
        assert abs(pi["d1"] - d1) < 1e-12


def test_d1_enters_lining_load_only():
    """Changing d1 must change the beam residual and nothing else."""
    inv = InverseD1(B01Config(seed=1, w_plate=8.4, **SMALL), _fake_gauges(), d1_0=0.6, noise=0.0)
    _, p0 = inv.losses()
    with torch.no_grad():
        inv.z.fill_(0.5)
    _, p1 = inv.losses()
    for k in ("eq", "c", "cpl", "data"):
        assert abs(p0[k] - p1[k]) <= 1e-12 * max(1.0, abs(p0[k])), k
    assert abs(p0["beam"] - p1["beam"]) > 1e-8 and 0.6 < p1["d1"] < 1.0


def test_gauges_and_noise(plate_file):
    gd0 = gauge_data(plate_file)
    assert gd0["snap_deg"] < 2.0 and len(gd0["theta"]) == len(GAUGES_DEG)
    assert np.allclose(gd0["N_obs"], gd0["N"]) and np.allclose(gd0["M_obs"], gd0["M"])
    a, b = gauge_data(plate_file, 0.05, 1), gauge_data(plate_file, 0.05, 1)
    c = gauge_data(plate_file, 0.05, 2)
    assert np.allclose(a["M_obs"], b["M_obs"]) and not np.allclose(a["M_obs"], c["M_obs"])
    # additive noise: the spread of (obs - true) should be noise * max|M|
    r = np.concatenate([gauge_data(plate_file, 0.1, s)["M_obs"] - gd0["M"] for s in range(400)])
    assert abs(r.std() / (0.1 * np.abs(gd0["M"]).max()) - 1) < 0.1


def test_d1_physics_matches_stage2_with_plaxis_gauges(plate_file):
    gd = gauge_data(plate_file)
    ref = Stage2(B01Config(seed=3, w_plate=8.4, d1=0.6, **SMALL))
    inv = InverseD1(B01Config(seed=3, w_plate=8.4, **SMALL), gd, d1_0=0.6, noise=0.0)
    inv.net.load_state_dict(ref.net.state_dict())
    inv.beam.load_state_dict(ref.beam.state_dict())
    _, pr = ref.losses()
    _, pi = inv.losses()
    for k in ("eq", "c", "beam", "cpl"):
        assert abs(pr[k] - pi[k]) <= 1e-12 * max(1.0, abs(pr[k])), k
    assert math.isfinite(pi["data"])
