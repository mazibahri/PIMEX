"""Train the forward model, stage 1 (excavation) or stage 2 (lining).

    python train.py --stage 1 --out runs/s1_seed0
    python train.py --stage 2 --w_plate 8.4 --out runs/s2_seed0

Writes model.pt, hist.json and log.txt into --out. No PLAXIS data is used here.
"""
import argparse
import json
import os
import time

import torch

from pinn import B01Config, Stage1, Stage2, train, _modules


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", type=int, choices=(1, 2), required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--adam", type=int, default=5000)
    p.add_argument("--lbfgs", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--npts", type=int, default=3000, help="collocation points per set (uniform and near-tunnel)")
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--depth", type=int, default=5)
    p.add_argument("--E", type=float, default=10_000.0, help="soil Young's modulus (kPa)")
    p.add_argument("--d1", type=float, default=0.6, help="deconfinement before the lining is installed")
    p.add_argument("--w_plate", type=float, default=0.0, help="lining self-weight (kN/m per m)")
    p.add_argument("--no_plate_factor", action="store_true", help="use EA, EI as given (no 1/(1-nu^2))")
    p.add_argument("--eq_len", type=float, default=17.0)
    p.add_argument("--init", default=None, help="model.pt to warm-start from")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()

    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    cfg = B01Config(seed=a.seed, n_uniform=a.npts, n_ring=a.npts, width=a.width, depth=a.depth,
                    E=a.E, d1=a.d1, w_plate=a.w_plate, eq_len=a.eq_len,
                    plate_plane_strain_factor=not a.no_plate_factor)
    model = (Stage1 if a.stage == 1 else Stage2)(cfg, device=a.device)
    if a.init:
        sd = torch.load(a.init, map_location=a.device)
        for k, v in _modules(model).items():
            v.load_state_dict(sd[k])

    logf = open(os.path.join(a.out, "log.txt"), "w")

    def log(s):
        print(s)
        logf.write(s + "\n"); logf.flush()

    t0 = time.time()
    hist = train(model, n_adam=a.adam, lr=a.lr, n_lbfgs=a.lbfgs, log_every=250, log=log,
                 ckpt=os.path.join(a.out, "model.pt"))
    torch.save({k: v.state_dict() for k, v in _modules(model).items()}, os.path.join(a.out, "model.pt"))
    with open(os.path.join(a.out, "hist.json"), "w") as f:
        json.dump({"args": vars(a), "cfg": cfg.__dict__, "minutes": (time.time() - t0) / 60,
                   "hist": hist}, f, default=str)
    log(f"done in {(time.time() - t0) / 60:.1f} min")
    logf.close()


if __name__ == "__main__":
    main()
