"""Check the discrete-adjoint gradients (CD and CMy at constant CL) and the analytic volume gradient
against central finite differences at an evaluated design.

By default two variables are tested: the z- and the y-variable with the largest |dCD/dp|.
Every finite-difference point is a full fixed-CL direct run (restarted from the design's solution).

Usage: SU2_NP=8 python scripts/fd_check.py --workdir runs/wing_body [--eval 0] [--h 0.1] [--vars 12,40]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import su2run as S  # noqa: E402
from driver import Problem, Evaluator  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default="runs/wing_body")
    ap.add_argument("--eval", type=int, default=0, help="design folder dsn_NNN with gradients")
    ap.add_argument("--h", type=float, default=0.1, help="finite-difference step (m of control-point displacement)")
    ap.add_argument("--vars", default=None, help="comma-separated optimiser-variable indices")
    a = ap.parse_args()
    P = Problem.load(a.workdir)
    base = os.path.join(P.runs, f"dsn_{a.eval:03d}")
    p0 = np.loadtxt(os.path.join(base, "p.txt"))
    gCD = np.loadtxt(os.path.join(base, "grad_CD.txt"))
    gCM = np.loadtxt(os.path.join(base, "grad_CMy.txt"))
    E = Evaluator(P, prefix=f"fd{a.eval:03d}")
    E.last_restart = f"../dsn_{a.eval:03d}/restart_flow.dat"
    E.last_aoa = float(S.read_meta(os.path.join(base, "flow.meta"))["AOA"])
    if a.vars:
        ks = [int(v) for v in a.vars.split(",")]
    else:
        ks = [max((k for k, v in enumerate(P.pvars) if v[0] == c), key=lambda k: abs(gCD[k])) for c in ("z", "y")]
    out = []
    for k in ks:
        e = np.zeros(P.np)
        e[k] = a.h
        rp, rm = E.primal(p0 + e), E.primal(p0 - e)
        row = dict(p_index=int(k), var=[str(v) for v in P.pvars[k]], at_eval=a.eval, h=a.h,
                   dCD_adj=float(gCD[k]), dCD_fd=float((rp["CD"] - rm["CD"]) / (2 * a.h)),
                   dCMy_adj=float(gCM[k]), dCMy_fd=float((rp["CMy"] - rm["CMy"]) / (2 * a.h)),
                   dVrel_analytic=float(P.dc_vol(p0)[k] / 100.0),
                   dVrel_fd=float((P.v_rel(p0 + e) - P.v_rel(p0 - e)) / (2 * a.h)))
        for q in ("CD", "CMy"):
            row[f"err_{q}_pct"] = 100 * abs(row[f"d{q}_adj"] - row[f"d{q}_fd"]) / abs(row[f"d{q}_fd"])
        row["err_V_pct"] = 100 * abs(row["dVrel_analytic"] - row["dVrel_fd"]) / abs(row["dVrel_fd"])
        out.append(row)
        print(json.dumps(row), flush=True)
    json.dump(out, open(os.path.join(P.runs, f"fd_check_{a.eval:03d}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
