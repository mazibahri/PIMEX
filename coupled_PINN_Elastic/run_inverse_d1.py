"""Inverse 2 (coupled): lining N, M at 5 gauges -> release d1 -> surface settlement. E is known.

    python run_inverse_d1.py --data data --init runs/s2_d04 --s1_basis runs/s1_seed0 \
        --d1_0 0.4 --noise 0.05 --seed 0 --out runs/invd1_d04_n5_s0

--init      stage-2 forward run trained at d1 = d1_0 (train.py --stage 2 --d1 0.4 ...). This is the
            warm start; do not use a run trained at the true d1.
--s1_basis  any stage-1 forward run with the same E. Stage 1 is linear in d1, so
            u1(d1) = (d1 / d1_b) * u1(d1_b) exactly, and the basis run carries no information
            about the true d1.

Only the gauge values are used in training. The full-ring N, M, Q and the lining-phase surface
settlement from PLAXIS are used afterwards to check the prediction.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pinn import B01Config, predict_ground, load_checkpoint
from inverse import InverseD1, gauge_data, train_inverse, GAUGES_DEG
from plaxis_io import read_plate, plate_unique, read_soil_displacements


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data")
    p.add_argument("--model", default="Elastic", help="file name prefix of the PLAXIS tables")
    p.add_argument("--init", required=True)
    p.add_argument("--s1_basis", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--d1_0", type=float, required=True, help="initial guess of d1")
    p.add_argument("--d1_true", type=float, default=0.6, help="value used in PLAXIS, for reporting only")
    p.add_argument("--noise", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--adam", type=int, default=3000)
    p.add_argument("--lbfgs", type=int, default=3000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr_z", type=float, default=1e-2, help="Adam learning rate for the d1 parameter")
    p.add_argument("--w_data0", type=float, default=0.1)
    p.add_argument("--use", default="NM", help="which gauge readings to use: N, M or NM")
    p.add_argument("--w_plate", type=float, default=8.4)
    p.add_argument("--npts", type=int, default=3000)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()

    torch.set_num_threads(a.threads)
    torch.set_default_dtype(torch.float64)
    os.makedirs(a.out, exist_ok=True)
    D = lambda s: os.path.join(a.data, f"{a.model}_{s}.TXT")

    # the warm start must match the assumptions of this run
    with open(os.path.join(a.init, "hist.json")) as f:
        cfg_init = json.load(f)["cfg"]
    for key, want in (("d1", a.d1_0), ("w_plate", a.w_plate)):
        if abs(cfg_init[key] - want) > 1e-9 * max(1.0, abs(want)):
            raise SystemExit(f"--init was trained with {key} = {cfg_init[key]}, expected {want}")

    gd = gauge_data(D("Plate_forces_TablePlateForces_LinPhase"), noise=a.noise, seed=a.seed)
    cfg = B01Config(seed=a.seed, n_uniform=a.npts, n_ring=a.npts, w_plate=a.w_plate, E=cfg_init["E"])
    m = InverseD1(cfg, gd, d1_0=a.d1_0, noise=a.noise, w_data0=a.w_data0,
                  use=tuple(a.use), device=a.device)
    sd = torch.load(os.path.join(a.init, "model.pt"), map_location=a.device)
    m.net.load_state_dict(sd["net"]); m.beam.load_state_dict(sd["beam"])

    logf = open(os.path.join(a.out, "log.txt"), "w")

    def log(s):
        print(s)
        logf.write(s + "\n"); logf.flush()

    log(f"gauges {GAUGES_DEG} deg (max snap {gd['snap_deg']:.2f} deg), use {a.use}, noise {a.noise}, "
        f"w_data {m.w_data:.4g}, d1_0 {a.d1_0}, device {a.device}")
    t0 = time.time()
    hist = train_inverse(m, a.adam, a.lr, a.lr_z, a.lbfgs, log, os.path.join(a.out, "model.pt"))
    torch.save({"net": m.net.state_dict(), "beam": m.beam.state_dict(), "z": m.z.detach().cpu(),
                "d1_0": a.d1_0}, os.path.join(a.out, "model.pt"))
    d1_id = float(m.d1.item())
    minutes = (time.time() - t0) / 60

    # evaluation on withheld PLAXIS data
    m.net.cpu(); m.beam.cpu()
    pf = plate_unique(read_plate(D("Plate_forces_TablePlateForces_LinPhase")), yc=cfg.yc)
    th = torch.tensor(pf.theta.values, requires_grad=True)
    b = m.beam_fields(th)
    N = b["N"].detach().numpy()
    M = -b["M"].detach().numpy()                     # back to the PLAXIS sign
    Q = (-b["M1"] / cfg.a).detach().numpy()
    rel = lambda p_, t_: float(np.linalg.norm(p_ - t_) / np.linalg.norm(t_) * 100)

    dL = read_soil_displacements(D("Soil_displacements_TableTotalDisplacements_LinPhase"))
    surf = dL[dL.y > -1e-9].sort_values("x")
    xs = surf.x.values
    m1, c1, _ = load_checkpoint(a.s1_basis, 1)
    if abs(c1.E - cfg.E) > 1e-6:
        raise SystemExit(f"stage-1 basis has E = {c1.E}, but the stage-2 run uses E = {cfg.E}")
    u1 = predict_ground(m1.net, xs, np.zeros_like(xs))[:, 1] * (d1_id / c1.d1)
    du2 = predict_ground(m.net, xs, np.zeros_like(xs))[:, 1]
    pred, ref = u1 + du2, surf.uy.values

    res = {
        "d1_true": a.d1_true, "d1_0": a.d1_0, "d1_identified": d1_id, "d1_error_abs": d1_id - a.d1_true,
        "noise": a.noise, "seed": a.seed, "use": a.use, "w_data": m.w_data, "gauges_deg": list(GAUGES_DEG),
        "minutes": minutes, "device": a.device, "adam": a.adam, "lbfgs": a.lbfgs, "lr": a.lr, "lr_z": a.lr_z,
        "init": a.init, "s1_basis": a.s1_basis, "s1_basis_d1": c1.d1,
        "lining_N_relL2_pct": rel(N, pf.N.values), "lining_M_relL2_pct": rel(M, pf.M.values),
        "lining_Q_relL2_pct": rel(Q, pf.Q.values),
        "settlement_relL2_pct": rel(pred, ref),
        "settlement_x0_mm": {"PLAXIS": float(ref[0] * 1e3), "PINN": float(pred[0] * 1e3)},
        "settlement_maxabs_err_mm": float(np.abs(pred - ref).max() * 1e3),
        "surface_x_m": xs.tolist(), "surface_uy_PLAXIS_m": ref.tolist(), "surface_uy_PINN_m": pred.tolist(),
        "d1_trajectory": [(h[0], h[1], h[3]["d1"]) for h in hist], "final_loss": hist[-1][3],
    }
    with open(os.path.join(a.out, "result.json"), "w") as f:
        json.dump(res, f, indent=1)
    log(json.dumps({k: v for k, v in res.items() if not isinstance(v, list)}))

    # figure: d1 during training, lining forces, settlement
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    tr = res["d1_trajectory"]
    n_adam = max([h[1] for h in tr if h[0] == "adam"] + [0]) + 1
    it = [h[1] for h in tr]               # L-BFGS iterations are already offset by n_adam
    ax[0].plot(it, [h[2] for h in tr], "r-", label=f"identified: {d1_id:.3f}")
    ax[0].axhline(a.d1_true, color="k", ls=":", label=f"PLAXIS: {a.d1_true}")
    ax[0].axvline(n_adam, color="0.6", lw=0.8)
    ax[0].set(xlabel="iteration (Adam, then L-BFGS)", ylabel="d1", title="Release factor during training")
    ax[0].legend(); ax[0].grid(alpha=.3)
    tdeg = np.rad2deg(pf.theta.values)
    ax[1].plot(tdeg, pf.M, "k-", label="PLAXIS M (withheld)")
    ax[1].plot(tdeg, M, "r--", label="PINN M")
    ax[1].plot(np.rad2deg(gd["theta"]), -gd["M_obs"], "bo", label="M gauges used")
    ax[1].set(xlabel="theta (deg)", ylabel="M (kNm/m, PLAXIS sign)", title="Lining bending moment")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    ax[2].plot(xs, ref * 1e3, "k-", lw=2, label="PLAXIS (withheld)")
    ax[2].plot(xs, pred * 1e3, "r--", lw=2, label="PINN from lining gauges")
    ax[2].set(xlabel="x (m)", ylabel="surface u_y (mm)", title="Lining phase: surface settlement")
    ax[2].legend(); ax[2].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "inverse_d1.png"), dpi=130); plt.close(fig)

    print(f"d1 = {d1_id:.4f} (error {d1_id - a.d1_true:+.4f}), settlement rel L2 "
          f"{res['settlement_relL2_pct']:.2f} %, {minutes:.1f} min")
    logf.close()


if __name__ == "__main__":
    main()
