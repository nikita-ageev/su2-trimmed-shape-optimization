"""First-order optimality (KKT) check on the accepted SLSQP iterations.

At an iterate p_k with gradients:  grad f = lambda grad c_eq + sum mu_i grad c_i  (active inequalities and bounds),
mu_i >= 0. The multipliers are found by sign-constrained least squares (scipy lsq_linear); the relative residual
||grad f - A mu|| / ||grad f|| measures stationarity. Scaling as in the driver (CD * 1e4, CMy * 1e3).

Usage: python scripts/kkt.py --workdir runs/wing_body
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.optimize import lsq_linear

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from driver import Problem, F_SCALE, C_SCALE  # noqa: E402

TOL_C = 2e-3


def kkt_at(P, ev):
    d = os.path.join(P.runs, f"dsn_{ev:03d}")
    p = np.loadtxt(os.path.join(d, "p.txt"))
    g = F_SCALE * np.loadtxt(os.path.join(d, "grad_CD.txt"))
    cols, names, lb = [C_SCALE * np.loadtxt(os.path.join(d, "grad_CMy.txt"))], ["CMy = 0"], [-np.inf]
    cv = P.c_vol(p)
    if cv < TOL_C:
        cols.append(P.dc_vol(p)); names.append("volume"); lb.append(0.0)
    nsm = 0
    if len(P.Lsm):
        csm, jsm = P.c_smooth(p), P.dc_smooth(p)
        for i in np.where(csm < TOL_C)[0]:
            cols.append(jsm[i]); names.append(f"smoothness #{i}"); lb.append(0.0); nsm += 1
    nb = 0
    for i, (lo, hi) in enumerate(P.bounds()):
        e = np.zeros(P.np)
        if p[i] <= lo + 1e-6:
            e[i] = 1.0
        elif p[i] >= hi - 1e-6:
            e[i] = -1.0
        else:
            continue
        cols.append(e); names.append(f"bound {P.pvars[i]}"); lb.append(0.0); nb += 1
    A = np.array(cols).T
    r = lsq_linear(A, g, bounds=(np.array(lb), np.full(len(lb), np.inf)))
    res = g - A @ r.x
    return dict(eval=ev, rel_residual=float(np.linalg.norm(res) / np.linalg.norm(g)), grad_norm=float(np.linalg.norm(g)),
                n_active_smooth=nsm, n_active_bounds=nb, vol_active=bool(cv < TOL_C), lam_CMy=float(r.x[0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default="runs/wing_body")
    a = ap.parse_args()
    P = Problem.load(a.workdir)
    it_file = os.path.join(P.runs, "opt_iterations.txt")
    its = [0] + ([int(l.split("eval")[1].split()[0]) for l in open(it_file)] if os.path.exists(it_file) else [])
    out = []
    for k, ev in enumerate(its):
        if os.path.exists(os.path.join(P.runs, f"dsn_{ev:03d}", "grad_CD.txt")):
            r = kkt_at(P, ev)
            r["iter"] = k
            out.append(r)
            print(f"iter {k:2d} eval {ev:3d}  KKT residual {r['rel_residual']:.3e}  active bounds {r['n_active_bounds']}"
                  f"  smoothness {r['n_active_smooth']}  volume {r['vol_active']}")
    json.dump(out, open(os.path.join(P.runs, "kkt.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
