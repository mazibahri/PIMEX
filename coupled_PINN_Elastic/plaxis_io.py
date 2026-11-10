"""Readers for PLAXIS 2D Output tables exported as tab-separated .txt.

PLAXIS sign convention: tension positive, compression negative (same as ours),
y upward, coordinates in m, stresses in kPa, forces in kN/m.
Each node appears once per element it belongs to, so node tables contain
duplicates; `unique_nodes` collapses them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def read_table(path: str) -> pd.DataFrame:
    with open(path, encoding="utf-8-sig") as f:
        lines = [ln.rstrip("\n\r") for ln in f if ln.strip()]
    header = [h.strip() for h in lines[0].split("\t")]
    rows = [ln.split("\t") for ln in lines[1:]]
    rows = [r for r in rows if len(r) == len(header)]
    df = pd.DataFrame(rows, columns=header)
    for c in df.columns:
        conv = pd.to_numeric(df[c].str.strip(), errors="coerce")
        if conv.notna().mean() > 0.99:
            df[c] = conv
        else:
            df[c] = df[c].str.strip()
    return df


def _rename(df, mapping):
    return df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})


def read_soil_displacements(path):
    df = _rename(read_table(path), {"Node": "node", "X [m]": "x", "Y [m]": "y",
                                    "u_x [m]": "ux", "u_y [m]": "uy"})
    return unique_nodes(df[["node", "x", "y", "ux", "uy"]])


def unique_nodes(df):
    return df.drop_duplicates(subset="node").sort_values("node").reset_index(drop=True)


def read_soil_stresses(path):
    df = _rename(read_table(path), {"Stress point": "sp", "X [m]": "x", "Y [m]": "y",
                                    "σ'_xx [kN/m²]": "sxx", "σ'_yy [kN/m²]": "syy",
                                    "σ'_zz [kN/m²]": "szz", "σ_xy [kN/m²]": "sxy"})
    cols = ["sp", "x", "y", "sxx", "syy", "szz", "sxy"]
    if "Status" in df.columns:
        df["status"] = df["Status"]
        cols.append("status")
    return df[cols].drop_duplicates(subset="sp").reset_index(drop=True)


def read_soil_strains(path):
    """Total strains -> Cartesian (exx, eyy, gxy engineering).
    PLAXIS lists sorted principal strains eps_1 <= eps_2 <= eps_3; the out-of-plane one is exactly 0
    (plane strain). 'Angle' is the direction (from x) of the algebraically LARGER in-plane principal
    strain. Verified on the elastic export: Hooke(strains) = stress change to 1e-12."""
    t = read_table(path)
    t = _rename(t, {"Stress point": "sp", "X [m]": "x", "Y [m]": "y", "Angle [°]": "ang"})
    e = t[["ε_1", "ε_2", "ε_3"]].to_numpy()
    z = np.argmin(np.abs(e), 1)
    ip = np.array([np.delete(row, k) for row, k in zip(e, z)])
    ea, eb = ip.max(1), ip.min(1)
    a = np.deg2rad(t["ang"].to_numpy())
    c, s = np.cos(a), np.sin(a)
    t["exx"] = ea * c**2 + eb * s**2
    t["eyy"] = ea * s**2 + eb * c**2
    t["gxy"] = -2.0 * (ea - eb) * s * c          # PLAXIS angle sense: verified via Hooke
    return t[["sp", "x", "y", "exx", "eyy", "gxy"]]


def read_plate(path):
    df = _rename(read_table(path), {"Node": "node", "X [m]": "x", "Y [m]": "y",
                                    "u_x [m]": "ux", "u_y [m]": "uy", "φ_z [°]": "phi",
                                    "N [kN/m]": "N", "Q [kN/m]": "Q", "M [kN m/m]": "M"})
    keep = [c for c in ["node", "x", "y", "ux", "uy", "phi", "N", "Q", "M"] if c in df.columns]
    df = df[keep].copy()
    return df


def plate_unique(df, xc=0.0, yc=-17.0):
    """Average duplicated element-end values; add polar angle theta (from +x, CCW)."""
    g = df.groupby("node", as_index=False).mean(numeric_only=True)
    g["theta"] = np.arctan2(g["y"] - yc, g["x"] - xc)
    return g.sort_values("theta").reset_index(drop=True)
