"""Inverse 1: identify E from 19 displacement sensors (excavation phase).

    python run_inverse_E.py --data data --E0 5000 --noise 0.05 --seed 0 --out runs/invE_E0_5000_n5_s0

E0 is the initial guess (kPa). --E_true is only used to report the error after training.
After training the model is evaluated against all withheld PLAXIS nodes of the excavation phase.
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

from pinn import B01Config, train, predict_ground
from inverse import InverseE, sensor_data
from plaxis_io import read_soil_displacements, read_soil_stresses


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data")
    p.add_argument("--model", default="Elastic", help="file name prefix of the PLAXIS tables")
    p.add_argument("--out", required=True)
    p.add_argument("--E0", type=float, default=5000.0, help="initial guess of E (kPa)")
    p.add_argument("--E_true", type=float, default=10000.0, help="value used in PLAXIS, for reporting only")
    p.add_argument("--noise", type=float, default=0.0, help="relative noise on the sensors, e.g. 0.05")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--adam", type=int, default=3000)
    p.add_argument("--lbfgs", type=int, default=3000)
    p.add_argument("--w_data", type=float, default=10.0)
    p.add_argument("--npts", type=int, default=3000)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()

    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    D = lambda s: os.path.join(a.data, f"{a.model}_{s}.TXT")
    f_disp = D("Soil_displacements_TableTotalDisplacements_ExcPhase")

    cfg = B01Config(seed=a.seed, n_uniform=a.npts, n_ring=a.npts, E=a.E0)   # cfg.E is not used by InverseE
    xy, comp, clean, obs, dmax = sensor_data(f_disp, cfg, noise=a.noise, seed=a.seed)
    m = InverseE(cfg, xy, comp, obs, E0=a.E0, w_data=a.w_data, device=a.device)

    logf = open(os.path.join(a.out, "log.txt"), "w")

    def log(s):
        print(s)
        logf.write(s + "\n"); logf.flush()

    log(f"{len(obs)} sensors, max snap distance {dmax:.3g} m, noise {a.noise}")
    t0 = time.time()
    hist = train(m, n_adam=a.adam, lr=1e-3, n_lbfgs=a.lbfgs, log_every=250, log=log)
    torch.save({"net": m.net.state_dict(), "q": m.q.detach().cpu(), "E0": a.E0},
               os.path.join(a.out, "model.pt"))

    # evaluation on the withheld PLAXIS data
    E_id = float(m.E.item())
    m.net.cpu()
    d = read_soil_displacements(f_disp)
    S = read_soil_stresses(D("Soil_stresses_TableEffectiveStresses_ExcPhase"))
    P = predict_ground(m.net, d.x.values, d.y.values)
    Q = predict_ground(m.net, S.x.values, S.y.values)
    near = np.hypot(d.x, d.y - cfg.yc).values < 3 * cfg.a
    nearS = np.hypot(S.x, S.y - cfg.yc).values < 3 * cfg.a
    rel = lambda p_, t_: float(np.linalg.norm(p_ - t_) / np.linalg.norm(t_))
    dsyy = S.syy.values - cfg.gamma * S.y.values
    res = {"E_true": a.E_true, "E0": a.E0, "E_identified": E_id,
           "E_error_pct": (E_id - a.E_true) / a.E_true * 100,
           "noise": a.noise, "seed": a.seed, "n_sensors": int(len(obs)),
           "minutes": (time.time() - t0) / 60,
           "uy_relL2_all": rel(P[:, 1], d.uy.values), "ux_relL2_all": rel(P[:, 0], d.ux.values),
           "uy_relL2_r<3R": rel(P[near, 1], d.uy.values[near]),
           "ux_relL2_r<3R": rel(P[near, 0], d.ux.values[near]),
           "dsyy_relL2_r<3R": rel(Q[nearS, 3], dsyy[nearS]),
           "E_trajectory": [(h[0], h[1], h[3]["E"]) for h in hist], "final_loss": hist[-1][3]}
    with open(os.path.join(a.out, "result.json"), "w") as f:
        json.dump(res, f, indent=1)
    log(json.dumps({k: v for k, v in res.items() if k != "E_trajectory"}))

    # figure: E during training, and the settlement trough against the withheld PLAXIS one
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    tr = res["E_trajectory"]
    n_adam = max([h[1] for h in tr if h[0] == "adam"] + [0]) + 1
    it = [h[1] if h[0] == "adam" else n_adam + h[1] for h in tr]
    ax[0].plot(it, np.array([h[2] for h in tr]) / 1e3, "r-", label=f"identified: {E_id / 1e3:.2f} MPa")
    ax[0].axhline(a.E_true / 1e3, color="k", ls=":", label=f"PLAXIS: {a.E_true / 1e3:.1f} MPa")
    ax[0].set(xlabel="iteration (Adam, then L-BFGS)", ylabel="E (MPa)", title="E during training")
    ax[0].legend(); ax[0].grid(alpha=.3)
    surf = d[d.y > -1e-9].sort_values("x")
    Ps = predict_ground(m.net, surf.x.values, surf.y.values)
    on_surf = (comp == 1) & (xy[:, 1] > -1)
    ax[1].plot(surf.x, surf.uy * 1e3, "k-", lw=2, label="PLAXIS (withheld)")
    ax[1].plot(surf.x, Ps[:, 1] * 1e3, "r--", lw=2, label="inverse PINN")
    ax[1].plot(xy[on_surf, 0], obs[on_surf] * 1e3, "bo", label=f"sensors ({a.noise * 100:.0f} % noise)")
    ax[1].set(xlabel="x (m)", ylabel="surface u_y (mm)", title="Excavation phase: surface settlement")
    ax[1].legend(); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "inverse_E.png"), dpi=130); plt.close(fig)
    print(f"E = {E_id / 1e3:.3f} MPa (error {res['E_error_pct']:+.2f} %), {res['minutes']:.1f} min")
    logf.close()


if __name__ == "__main__":
    main()
