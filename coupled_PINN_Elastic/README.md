# Coupled ground–lining PINN, linear elastic (benchmark 01)

A physics-informed neural network (PINN) for a circular tunnel in linear elastic ground with a bonded,
curved Euler–Bernoulli lining, compared against a PLAXIS 2D model of the same problem. The repository
also contains two inverse problems:

1. **E from displacements**: find the soil Young's modulus from 19 displacement sensors (surface
   settlement and tunnel convergence).
2. **d1 from lining forces (coupled)**: find the release factor before lining installation from N and M
   at 5 lining gauges, then predict the surface settlement.

Everything runs from this folder (`coupled_PINN_Elastic/`). The `data/` folder is empty on purpose: the PLAXIS reference results
are not included. You can ask me for them, or build your own PLAXIS model (see [Data](#data)).

Part of [PIMEX](../README.md). Author: Maziyar Bahri, Durham University (maziyar.bahri@durham.ac.uk).
Licence: MIT (see `LICENSE`).

---

## The problem

| | |
|---|---|
| Model | plane strain, half model (symmetry about x = 0) |
| Domain | x ∈ [0, 35] m, y ∈ [−30, 0] m, y up, ground surface at y = 0 |
| Tunnel | radius R = 2.5 m, centre at (0, −17) m |
| Soil | dry, linear elastic: E = 10 MPa, ν = 0.33, γ = 15 kN/m³, K0 = 0.6 |
| Lining | plate: EA = 14·10⁶ kN/m, EI = 143·10³ kNm²/m, ν = 0.15, bonded to the soil |
| Phases | initial (K0) → excavation with 60 % release, no lining → lining activated, release completed |

Both stages are written as increments on top of the geostatic state σ0 = diag(K0·γ·y, γ·y). The geostatic
state is self-equilibrated, so the increments carry no body force.

- **Stage 1 (excavation):** div Δσ = 0, Δσ = C:ε(Δu), and on the tunnel wall Δσ·e_r = −d1·σ0·e_r (d1 = 0.6).
- **Stage 2 (lining):** same soil equations. The soil displacement at the wall equals the lining
  displacement. The lining carries p = ((1 − d1)·σ0 + Δσ)·e_r, plus its self-weight if any, and follows the
  curved beam equations

  N′ + M′/a + a·p_θ = 0,  N − M″/a − a·p_r = 0,  N = EA(u_θ′ + u_r)/a,  M = EI(u_θ′ − u_r″)/a².

Because the soil is linear elastic, stage 2 does not depend on stage 1. The two stages are trained
separately and added together.

Network details:

- **Ground network:** (x, y) → (u_x, u_y, σ_xx, σ_yy, σ_xy).
- **Boundary conditions:** all straight-boundary conditions are imposed exactly (hard constraints).
- **Symmetry:** exact, because the inputs are even in x.
- **Near-tunnel features:** a few (a/r)ⁿ features are added to the inputs.
- **Lining network:** θ → (u_θ, N, M). The axial law is imposed exactly and the bending law is penalised.

The full set of equations is in the docstring at the top of `pinn.py`.

### Conventions

- **Units:** m, kPa, kN/m and kNm/m.
- **Stress sign:** tension is positive, the same as PLAXIS.
- **Angle:** θ is measured anticlockwise from +x, so the crown is at +90° and the invert at −90°.
- **Bending moment:** M in this code has the opposite sign to the PLAXIS plate M. `compare.py` and
  `run_inverse_d1.py` flip it before comparing.
- **Plate stiffness:** by default EA and EI are divided by (1 − ν²), as PLAXIS plates are in plane
  strain. Switch this off with `--no_plate_factor`.

---

## Files

| File | What it does |
|---|---|
| `pinn.py` | configuration, networks, residuals, Stage1 / Stage2, training loop |
| `plaxis_io.py` | readers for PLAXIS Output tables (tab-separated .txt) |
| `inverse.py` | inverse models: `InverseE` (stage 1) and `InverseD1` (coupled stage 2), sensor/gauge data |
| `train.py` | train the forward model, one stage at a time |
| `check_physics.py` | residuals and lining checks on fresh points (no PLAXIS needed) |
| `compare.py` | forward model vs PLAXIS: metrics + figures |
| `check_plaxis_export.py` | sanity checks on a PLAXIS export, run this first with your own model |
| `run_inverse_E.py` | inverse 1: E from displacement sensors |
| `run_inverse_d1.py` | inverse 2: d1 from lining gauges → surface settlement |
| `tests/` | unit tests (`pytest`) |
| `jobs/run_all.sh` | the whole pipeline in one go |
| `jobs/slurm_forward.sh` | template for a SLURM cluster |
| `data/` | put the PLAXIS tables here (empty) |

---

## Installation

Python ≥ 3.9 and PyTorch ≥ 2.0. A GPU helps but is not required; everything is in float64.

```bash
python -m venv .venv && source .venv/bin/activate     # optional
pip install -r requirements.txt
python -m pytest -q tests                             # data-dependent tests are skipped if data/ is empty
```

---

## Data

The PINN itself does not need any data: the forward model is trained from the physics alone.
PLAXIS is used only to check the results (`compare.py`) and as synthetic "measurements" for the inverse
problems.

You can get the reference tables in two ways:

**1. Ask me.** I can share the PLAXIS 2D export I used. Email me at maziyar.bahri@durham.ac.uk and copy
the files into `data/`.

**2. Build your own PLAXIS 2D model.** Use the parameters in [The problem](#the-problem). Things that matter:

- **Geometry:** a half model with the tunnel centre on the left boundary.
- **Fixities:** keep the PLAXIS default fixities (normally fixed on the sides, fully fixed at the
  bottom, free surface).
- **Soil:** linear elastic and dry, with no water table. Effective and total stresses are then the same.
- **Initial phase:** K0 procedure with K0 = 0.6.
- **Excavation phase:** deactivate the tunnel soil and apply 60 % of the release, for example by stopping
  the phase at ΣMstage = 0.6. Do not activate the lining.
- **Lining phase:** activate the plate and complete the release.
- **Interface:** the lining must be bonded. Delete the tunnel interface, or check that the plate nodes
  coincide with the soil nodes. `check_plaxis_export.py` reports this. With an elastic interface the
  excavation phase is unaffected, but lining N and M are no longer comparable.
- **Self-weight:** our model's plate material had w = 8.4 kN/m/m (a value from a PLAXIS tutorial). Pass
  the same value with `--w_plate`, or set w = 0 in PLAXIS and use `--w_plate 0`.

Export these tables from PLAXIS Output (Tables → right click → export to text). Name them as follows,
with the same prefix (default `Elastic`, change with `--model`):

```
data/Elastic_Soil_displacements_TableTotalDisplacements_ExcPhase.TXT    # nodes: Node, X [m], Y [m], u_x [m], u_y [m]
data/Elastic_Soil_displacements_TableTotalDisplacements_LinPhase.TXT
data/Elastic_Soil_stresses_TableEffectiveStresses_ExcPhase.TXT          # stress points: Stress point, X [m], Y [m],
                                                                        #   σ'_xx, σ'_yy, σ'_zz, σ_xy [kN/m²]
data/Elastic_Plate_forces_TablePlateForces_LinPhase.TXT                 # Node, X [m], Y [m], N [kN/m], Q [kN/m], M [kN m/m]
data/Elastic_Plate_displacements_TablePlateDisplacements_LinPhase.TXT   # Node, X [m], Y [m], u_x [m], u_y [m]
```

The readers in `plaxis_io.py` find columns by these header names. If your PLAXIS version writes
different headers, change the dictionaries there.

If your geometry or material values are different, change the defaults in `B01Config` (`pinn.py`) and
the matching arguments of `check_plaxis_export.py`. The sensor positions in `inverse.py` assume this
geometry.

---

## Running

Everything below is run from this folder. The quickest way is

```bash
QUICK=1 bash jobs/run_all.sh     # a few minutes on a CPU, only to see that everything runs
bash jobs/run_all.sh             # full run
```

or step by step:

```bash
# 0. check the PLAXIS export
python check_plaxis_export.py --data data

# 1. forward model (no data used)
python train.py --stage 1 --out runs/s1
python train.py --stage 2 --w_plate 8.4 --out runs/s2
python check_physics.py --s1 runs/s1 --s2 runs/s2
python compare.py --s1 runs/s1 --s2 runs/s2 --data data --out runs/compare

# 2. inverse 1: E from 19 displacement sensors (start from E0 = 5 MPa, 5 % noise)
python run_inverse_E.py --data data --E0 5000 --noise 0.05 --seed 0 --out runs/inv_E

# 3. inverse 2: d1 from lining N, M (warm start: a stage-2 run at d1 = 0.4, not the true 0.6)
python train.py --stage 2 --w_plate 8.4 --d1 0.4 --out runs/s2_d04
python run_inverse_d1.py --data data --init runs/s2_d04 --s1_basis runs/s1 \
    --d1_0 0.4 --noise 0.05 --seed 0 --out runs/inv_d1
```

Each run folder gets a `log.txt`, the checkpoint `model.pt`, a `hist.json` / `result.json` and figures.
`python <script>.py -h` lists all options.

On a laptop CPU one inverse run took about 45 minutes with the default iterations; a GPU is faster.

---

## What to check

A small loss does not mean a correct solution. Before trusting a run, look at:

- **Global vertical force balance** (`compare.py`, `global_Fy_check`). Integrating Δσ_yy near the bottom
  boundary must give the released weight d1·γ·πR²/2. This is how we found that scaling the equilibrium
  residual by the tunnel radius lets a small, smooth residual act as a body force; it is now scaled by
  the tunnel depth (`eq_len`).
- **Residuals on fresh points** (`check_physics.py`): equilibrium, constitutive, wall traction, beam
  equations and soil–lining compatibility.
- **Several seeds** (`--seed`), and for the inverse problems several noise realisations.

In our runs the forward model matched PLAXIS well. Near the tunnel (r < 3R), displacement errors were
about 1–1.5 % (relative L2) and lining N errors were below 0.1 %. The lining bending moment differed by about 2 %, and this did
not go away with more training. It is probably a difference between the Euler–Bernoulli ring here and the
PLAXIS plate element, which is not resolved. These are comparisons with a numerical model of the same
equations (verification), not validation against field data.

---

## Limitations

- Linear elastic soil only, a single geometry and a fully bonded lining.
- The inverse problems identify one parameter at a time; the others are assumed known.
- Synthetic measurements come from PLAXIS. They are therefore not generated by the same solver as the
  PINN, but they are still noise-free model data rather than monitoring data.

---

## Funding

This work is part of PIMEX, funded by the European Union under the Horizon Europe Marie Skłodowska-Curie
Actions programme, grant agreement 101276407.
