"""Compare trained models (stage 1, optionally stage 2) with the PLAXIS export.

    python compare.py --s1 runs/s1_seed0 --s2 runs/s2_seed0 --data data --out results/forward

PLAXIS is only read here; the forward models never see it during training.
Writes metrics.json and two figures into --out.
"""
import argparse, json, os, math
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

from pinn import predict_ground, load_checkpoint
from plaxis_io import (read_soil_displacements, read_soil_stresses, read_plate, plate_unique)

torch.set_default_dtype(torch.float64)
_trapz = getattr(np, "trapezoid", None) or np.trapz     # numpy < 2 has only trapz


def rel_l2(p, t):
    return float(np.linalg.norm(p - t) / np.linalg.norm(t))


def load_model(path, stage):
    return load_checkpoint(path, stage)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s1", required=True); ap.add_argument("--s2", default=None)
    ap.add_argument("--data", default="data"); ap.add_argument("--out", default="results")
    ap.add_argument("--model", default="Elastic", help="file name prefix of the PLAXIS tables")
    A = ap.parse_args()
    os.makedirs(A.out, exist_ok=True)
    D = lambda s: os.path.join(A.data, f"{A.model}_{s}.TXT")
    m1, cfg, h1 = load_model(A.s1, 1)
    a, yc = cfg.a, cfg.yc
    res = {"stage1_minutes": h1["minutes"], "stage1_final_loss": h1["hist"][-1][2:]}

    # ---------------- stage 1 vs PLAXIS excavation phase
    dE = read_soil_displacements(D("Soil_displacements_TableTotalDisplacements_ExcPhase"))
    sE = read_soil_stresses(D("Soil_stresses_TableEffectiveStresses_ExcPhase"))
    P1 = predict_ground(m1.net, dE.x.values, dE.y.values)
    rN = np.hypot(dE.x, dE.y - yc).values; near = rN < 3 * a
    S = {}
    for k, j in [("ux", 0), ("uy", 1)]:
        S[f"{k}_relL2_all"] = rel_l2(P1[:, j], dE[k].values)
        S[f"{k}_relL2_r<3R"] = rel_l2(P1[near, j], dE[k].values[near])
        S[f"{k}_maxabs_mm"] = float(np.abs(P1[:, j] - dE[k].values).max() * 1e3)
    # stresses: compare increments ds = sigma_PLAXIS - sigma0
    Q1 = predict_ground(m1.net, sE.x.values, sE.y.values)
    s0xx, s0yy = cfg.K0 * cfg.gamma * sE.y.values, cfg.gamma * sE.y.values
    rS = np.hypot(sE.x, sE.y - yc).values; nearS = rS < 3 * a
    for k, j, s0 in [("sxx", 2, s0xx), ("syy", 3, s0yy), ("sxy", 4, 0 * s0xx)]:
        dP = sE[k].values - s0
        S[f"d{k}_relL2_all"] = rel_l2(Q1[:, j], dP)
        S[f"d{k}_relL2_r<3R"] = rel_l2(Q1[nearS, j], dP[nearS])
        S[f"{k}_total_relL2_r<3R"] = rel_l2(Q1[nearS, j] + s0[nearS], sE[k].values[nearS])
    # key points
    def at(df, x, y):
        i = cKDTree(np.c_[df.x, df.y]).query([x, y])[1]; return i
    kp = {"crown": (0, yc + a), "invert": (0, yc - a), "spring": (a, yc), "surface_x0": (0, 0)}
    S["key_points_mm"] = {n: {"PLAXIS_uy": float(dE.uy.values[at(dE, *xy)] * 1e3),
                              "PINN_uy": float(P1[at(dE, *xy), 1] * 1e3),
                              "PLAXIS_ux": float(dE.ux.values[at(dE, *xy)] * 1e3),
                              "PINN_ux": float(P1[at(dE, *xy), 0] * 1e3)} for n, xy in kp.items()}
    surf = dE[dE.y > -1e-9].sort_values("x"); Ps = predict_ground(m1.net, surf.x.values, surf.y.values)
    S["surface_uy_relL2"] = rel_l2(Ps[:, 1], surf.uy.values)
    # global vertical equilibrium: int dsyy dx near the bottom must equal the net
    # upward release force d1*gamma*pi*a^2/2 (half model)
    xm = np.linspace(0, cfg.L, 701)
    Pb = predict_ground(m1.net, xm, np.full_like(xm, -cfg.Hb + 0.5))
    S["global_Fy_check_kN_per_m"] = {"expected": cfg.d1 * cfg.gamma * math.pi * a**2 / 2,
                                     "PINN_int_dsyy_bottom": float(_trapz(Pb[:, 3], xm))}
    res["stage1_vs_PLAXIS_Exc"] = S

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(surf.x, surf.uy * 1e3, "k-", label="PLAXIS"); ax[0].plot(surf.x, Ps[:, 1] * 1e3, "r--", label="PINN")
    ax[0].set(xlabel="x (m)", ylabel="surface u_y (mm)", title="Excavation phase: surface settlement"); ax[0].legend()
    w = dE[np.abs(rN - a) < 1e-6].copy(); w["th"] = np.arctan2(w.y - yc, w.x); w = w.sort_values("th")
    Pw = predict_ground(m1.net, w.x.values, w.y.values)
    ax[1].plot(np.rad2deg(w.th), w.ux * 1e3, "k-", label="PLAXIS u_x"); ax[1].plot(np.rad2deg(w.th), Pw[:, 0] * 1e3, "r--", label="PINN u_x")
    ax[1].plot(np.rad2deg(w.th), w.uy * 1e3, "b-", label="PLAXIS u_y"); ax[1].plot(np.rad2deg(w.th), Pw[:, 1] * 1e3, "m--", label="PINN u_y")
    ax[1].set(xlabel="theta (deg, from springline)", ylabel="wall displacement (mm)", title="Excavation phase: tunnel wall"); ax[1].legend(fontsize=8)
    tc = ax[2].tricontourf(dE.x, dE.y, (P1[:, 1] - dE.uy.values) * 1e3, 30, cmap="RdBu_r")
    plt.colorbar(tc, ax=ax[2], label="PINN - PLAXIS u_y (mm)"); ax[2].set_aspect("equal"); ax[2].set_title("Excavation phase: u_y error")
    fig.tight_layout(); fig.savefig(os.path.join(A.out, "stage1_vs_plaxis.png"), dpi=130); plt.close(fig)

    if A.s2:
        m2, cfg2, h2 = load_model(A.s2, 2)
        res["stage2_minutes"] = h2["minutes"]; res["stage2_final_loss"] = h2["hist"][-1][2:]
        dL = read_soil_displacements(D("Soil_displacements_TableTotalDisplacements_LinPhase"))
        P2 = predict_ground(m2.net, dL.x.values, dL.y.values)
        P1L = predict_ground(m1.net, dL.x.values, dL.y.values)
        Ptot = P1L + P2
        T = {}
        rL = np.hypot(dL.x, dL.y - yc).values; nearL = rL < 3 * a
        iE = cKDTree(np.c_[dE.x, dE.y]).query(np.c_[dL.x, dL.y])[1]
        for k, j in [("ux", 0), ("uy", 1)]:
            T[f"total_{k}_relL2_all"] = rel_l2(Ptot[:, j], dL[k].values)
            T[f"total_{k}_relL2_r<3R"] = rel_l2(Ptot[nearL, j], dL[k].values[nearL])
            inc = dL[k].values - dE[k].values[iE]
            T[f"incr_{k}_relL2_all"] = rel_l2(P2[:, j], inc)
            T[f"incr_{k}_relL2_r<3R"] = rel_l2(P2[nearL, j], inc[nearL])
        # lining
        pf = plate_unique(read_plate(D("Plate_forces_TablePlateForces_LinPhase")))
        pu = plate_unique(read_plate(D("Plate_displacements_TablePlateDisplacements_LinPhase")))
        th = torch.tensor(pf.theta.values, requires_grad=True)
        b = m2.beam_fields(th)
        N = b["N"].detach().numpy(); M = -b["M"].detach().numpy()   # PLAXIS M sign = -ours
        T["N_relL2"] = rel_l2(N, pf.N.values); T["M_relL2"] = rel_l2(M, pf.M.values)
        T["N_maxerr_pct_of_max"] = float(np.abs(N - pf.N.values).max() / np.abs(pf.N.values).max() * 100)
        T["M_maxerr_pct_of_max"] = float(np.abs(M - pf.M.values).max() / np.abs(pf.M.values).max() * 100)
        for n, t in [("crown", math.pi / 2), ("spring", 0.0), ("invert", -math.pi / 2)]:
            T[f"N_{n}"] = {"PLAXIS": float(np.interp(t, pf.theta, pf.N)), "PINN": float(np.interp(t, pf.theta, N))}
            T[f"M_{n}"] = {"PLAXIS": float(np.interp(t, pf.theta, pf.M)), "PINN": float(np.interp(t, pf.theta, M))}
        # lining total displacement: stage1 soil at wall + beam increment
        thu = torch.tensor(pu.theta.values, requires_grad=True); bu = m2.beam_fields(thu)
        ur, ut = bu["ur"].detach().numpy(), bu["ut"].detach().numpy(); c_, s_ = np.cos(pu.theta.values), np.sin(pu.theta.values)
        P1w = predict_ground(m1.net, pu.x.values, pu.y.values)
        uxL = P1w[:, 0] + ur * c_ - ut * s_; uyL = P1w[:, 1] + ur * s_ + ut * c_
        T["lining_total_ux_relL2"] = rel_l2(uxL, pu.ux.values); T["lining_total_uy_relL2"] = rel_l2(uyL, pu.uy.values)
        # coupling residual: soil (stage 2) vs beam displacement at wall
        P2w = predict_ground(m2.net, pu.x.values, pu.y.values)
        T["coupling_max_mismatch_mm"] = float(max(np.abs(P2w[:, 0] - (ur * c_ - ut * s_)).max(), np.abs(P2w[:, 1] - (ur * s_ + ut * c_)).max()) * 1e3)
        xm = np.linspace(0, cfg2.L, 701)
        Pb2 = predict_ground(m2.net, xm, np.full_like(xm, -cfg2.Hb + 0.5))
        T["global_Fy_check_kN_per_m"] = {"expected": (1 - cfg2.d1) * cfg2.gamma * math.pi * a**2 / 2 - cfg2.w_plate * math.pi * a,
                                         "PINN_int_dsyy_bottom": float(_trapz(Pb2[:, 3], xm))}
        res["stage2_vs_PLAXIS_Lin"] = T

        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        tdeg = np.rad2deg(pf.theta.values)
        ax[0].plot(tdeg, pf.N, "k-", label="PLAXIS"); ax[0].plot(tdeg, N, "r--", label="PINN"); ax[0].set(xlabel="theta (deg)", ylabel="N (kN/m)", title="Lining axial force"); ax[0].legend()
        ax[1].plot(tdeg, pf.M, "k-", label="PLAXIS"); ax[1].plot(tdeg, M, "r--", label="PINN"); ax[1].set(xlabel="theta (deg)", ylabel="M (kNm/m, PLAXIS sign)", title="Lining bending moment"); ax[1].legend()
        surfL = dL[dL.y > -1e-9].sort_values("x"); PsL = predict_ground(m1.net, surfL.x.values, surfL.y.values) + predict_ground(m2.net, surfL.x.values, surfL.y.values)
        ax[2].plot(surfL.x, surfL.uy * 1e3, "k-", label="PLAXIS"); ax[2].plot(surfL.x, PsL[:, 1] * 1e3, "r--", label="PINN")
        ax[2].set(xlabel="x (m)", ylabel="surface u_y (mm)", title="Lining phase: surface settlement"); ax[2].legend()
        T["surface_uy_relL2_total"] = rel_l2(PsL[:, 1], surfL.uy.values)
        fig.tight_layout(); fig.savefig(os.path.join(A.out, "stage2_vs_plaxis.png"), dpi=130); plt.close(fig)

    json.dump(res, open(os.path.join(A.out, "metrics.json"), "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
