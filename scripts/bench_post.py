"""Figures for the axisymmetric benchmark: radius r(x) before/after, dr, Cp, optimisation history.

Usage: python scripts/bench_post.py --workdir runs/axi [--out figures]
"""
import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_axi as BA  # noqa: E402

GAM, RGAS, PINF, TINF, MINF = 1.4, 287.058, 101325.0, 288.15, 1.8


def cp_of(d):
    a = np.genfromtxt(os.path.join(d, "surface_flow.csv"), delimiter=",", names=True)
    v2 = (a["Momentum_x"] ** 2 + a["Momentum_y"] ** 2) / a["Density"] ** 2
    p = (GAM - 1) * (a["Energy"] - 0.5 * a["Density"] * v2)
    rinf = PINF / (RGAS * TINF)
    vinf = MINF * np.sqrt(GAM * RGAS * TINF)
    cp = (p - PINF) / (0.5 * rinf * vinf**2)
    s = np.argsort(a["x"])
    return a["x"][s], cp[s]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default="runs/axi")
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    wd, out = os.path.abspath(a.workdir), os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    res = json.load(open(os.path.join(wd, "result.json")))
    x = np.loadtxt(os.path.join(wd, "x_opt.txt"))
    body = BA.Body(os.path.join(wd, "sh_axi.su2"))
    xy0, y1 = body.xy0, body.radius(x)
    nodes = np.unique(body.seg)
    o = nodes[np.argsort(xy0[nodes, 0])]

    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 130})
    fig, ax = plt.subplots(3, 1, figsize=(9, 8), sharex=True, gridspec_kw=dict(height_ratios=[1.4, 1, 1.2]))
    ax[0].plot(xy0[o, 0], xy0[o, 1], color="#4a6fa5", label="Sears-Haack (baseline)")
    ax[0].plot(xy0[o, 0], y1[o], color="#d1495b", label="discrete adjoint + SLSQP")
    ax[0].set_ylabel("r / L"); ax[0].legend(fontsize=8)
    ax[0].set_title(f"Body of revolution, M = 1.8, Euler: CD {res['CD0']:.5f} -> {res['CD_opt']:.5f} "
                    f"({res['dCD_pct']:+.2f} %), V/V0 = {res['V_rel']:.5f}", fontsize=10)
    ax[1].plot(xy0[o, 0], (y1[o] - xy0[o, 1]) / BA.RMAX * 100, color="#d1495b")
    ax[1].axhline(0, color="k", lw=0.6); ax[1].set_ylabel("dr, % of R_max")
    x0c, cp0 = cp_of(os.path.join(wd, "e_000"))
    x1c, cp1 = cp_of(os.path.join(wd, f"e_{res['final_eval']:03d}"))
    ax[2].plot(x0c, cp0, color="#4a6fa5", lw=1, label="baseline")
    ax[2].plot(x1c, cp1, color="#d1495b", lw=1, label="optimised")
    ax[2].set_ylabel("Cp"); ax[2].set_xlabel("x / L"); ax[2].legend(fontsize=8); ax[2].set_ylim(-0.25, 0.3)
    fig.tight_layout(); fig.savefig(os.path.join(out, "bench_axi_shape.png")); plt.close(fig)

    h = np.genfromtxt(os.path.join(wd, "history_opt.csv"), delimiter=",", names=True)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ok = h["V_rel"] >= 1 - 2e-6
    ax.plot(h["eval"], h["CD"] / res["CD0"] * 100 - 100, ".", color="#bbbbbb", label="all solver calls (incl. line search)")
    best = np.minimum.accumulate(np.where(ok, h["CD"], np.inf))
    ax.plot(h["eval"], best / res["CD0"] * 100 - 100, color="#d1495b", label="best feasible")
    ax.set_xlabel("solver call"); ax.set_ylabel("dCD, %"); ax.legend(fontsize=8)
    ax.set_title("Axisymmetric benchmark: optimisation history", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(out, "bench_axi_history.png")); plt.close(fig)
    print("figures written to", out)


if __name__ == "__main__":
    main()
