"""Physics check of trained models on fresh points that were not used in training.

Reports equilibrium and constitutive residuals, the wall traction error (stage 1), and beam
equilibrium, bending law and soil-lining compatibility at the wall (stage 2).
A small loss is not enough on its own - look at these numbers and at the global force check
in compare.py before trusting a run.

    python check_physics.py --s1 runs/s1_seed0 --s2 runs/s2_seed0
"""
import argparse, json, math
import numpy as np
import torch
from pinn import load_checkpoint as load_model, ground_residuals, interior_points, wall_points, polar_traction, sigma0

torch.set_default_dtype(torch.float64)
ap = argparse.ArgumentParser(); ap.add_argument("--s1"); ap.add_argument("--s2"); A = ap.parse_args()
out = {}
for st, path in [(1, A.s1), (2, A.s2)]:
    if not path:
        continue
    m, cfg, _ = load_model(path, st)
    cfg.n_uniform = cfg.n_ring = 4000
    x, y = interior_points(cfg, np.random.default_rng(12345), "cpu", torch.float64)
    r_eq, r_c = ground_residuals(m.net, x, y)
    R = {"eq_rms_kPa_per_m": float(r_eq.pow(2).mean().sqrt()), "eq_max_kPa_per_m": float(r_eq.abs().max()),
         "const_rms_kPa": float(r_c.pow(2).mean().sqrt()), "const_max_kPa": float(r_c.abs().max()),
         "s_ref_kPa": m.s_ref, "body_force_equiv_vs_gamma_pct": float(r_eq.pow(2).mean().sqrt() / cfg.gamma * 100)}
    xw, yw, th = wall_points(cfg, 997, "cpu", torch.float64)
    xw.requires_grad_(True); yw.requires_grad_(True)
    if st == 1:
        _, _, sxx, syy, sxy = m.net(xw, yw)
        tr, tt = polar_traction(sxx, syy, sxy, th)
        s0xx, s0yy = sigma0(cfg, yw)
        tr0, tt0 = polar_traction(s0xx, s0yy, 0 * s0xx, th)
        R["wall_traction_err_max_kPa"] = float(max((tr + cfg.d1 * tr0).abs().max(), (tt + cfg.d1 * tt0).abs().max()))
        R["wall_traction_ref_kPa"] = float((cfg.d1 * tr0).abs().max())
    else:
        th = th.clone().requires_grad_(True)
        ux, uy, sxx, syy, sxy = m.net(xw, yw)
        s0xx, s0yy = sigma0(cfg, yw)
        tr, tt = polar_traction(sxx + (1 - cfg.d1) * s0xx, syy + (1 - cfg.d1) * s0yy, sxy, th)
        tr = tr - cfg.w_plate * torch.sin(th); tt = tt - cfg.w_plate * torch.cos(th)
        b = m.beam_fields(th)
        r1 = b["N1"] + b["M1"] / cfg.a + cfg.a * tt
        r2 = b["N"] - b["M2"] / cfg.a - cfg.a * tr
        r4 = b["M"] - cfg.EI_eff * b["kappa"]
        R["beam_eq_max_kN_per_m"] = float(max(r1.abs().max(), r2.abs().max()))
        R["beam_bending_law_max_kNm"] = float(r4.abs().max())
        c, s = torch.cos(th), torch.sin(th)
        R["coupling_max_mm"] = float(max((ux * c + uy * s - b["ur"]).abs().max(), (-ux * s + uy * c - b["ut"]).abs().max()) * 1e3)
        R["N_max_kN_per_m"] = float(b["N"].abs().max()); R["M_max_kNm"] = float(b["M"].abs().max())
    out[f"stage{st}"] = R
print(json.dumps(out, indent=2))
