#!/usr/bin/env bash
# Full pipeline, run from the repository root:   bash jobs/run_all.sh
# Quick smoke test (a few minutes on a CPU):    QUICK=1 bash jobs/run_all.sh
# PLAXIS tables are expected in ./data (or set DATA=/path/to/tables).
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PYTHON:-python}
DATA=${DATA:-data}
SEED=${SEED:-0}
W_PLATE=${W_PLATE:-8.4}          # lining self-weight used in the PLAXIS model (kN/m per m)

if [[ "${QUICK:-0}" == "1" ]]; then
    FWD="--adam 200 --lbfgs 100 --npts 500"; INV="--adam 100 --lbfgs 100 --npts 500"; TAG=quick_
else
    FWD="--adam 5000 --lbfgs 5000";          INV="--adam 3000 --lbfgs 3000";          TAG=""
fi
R=runs/${TAG}seed${SEED}

echo "== tests";                      $PY -m pytest -q tests --data "$DATA"
echo "== check PLAXIS export";        $PY check_plaxis_export.py --data "$DATA"

echo "== forward, stage 1";           $PY train.py --stage 1 --seed $SEED $FWD --out $R/s1
echo "== forward, stage 2";           $PY train.py --stage 2 --seed $SEED $FWD --w_plate $W_PLATE --out $R/s2
echo "== physics check";              $PY check_physics.py --s1 $R/s1 --s2 $R/s2 | tee $R/physics_check.json
echo "== compare with PLAXIS";        $PY compare.py --s1 $R/s1 --s2 $R/s2 --data "$DATA" --out $R/compare

echo "== inverse E (5 % noise)"
$PY run_inverse_E.py --data "$DATA" --E0 5000 --noise 0.05 --seed $SEED $INV --out $R/inv_E

echo "== inverse d1: warm start at d1 = 0.4"
$PY train.py --stage 2 --seed $SEED $FWD --w_plate $W_PLATE --d1 0.4 --out $R/s2_d04
$PY run_inverse_d1.py --data "$DATA" --init $R/s2_d04 --s1_basis $R/s1 --d1_0 0.4 \
    --noise 0.05 --seed $SEED --w_plate $W_PLATE $INV --out $R/inv_d1

echo "done; results in $R"
