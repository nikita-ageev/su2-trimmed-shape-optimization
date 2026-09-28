"""Post-processing of an optimisation run: PNG figures (shape, sections, area distribution, Cp, Mach field,
convergence, optimisation history).

Usage: python scripts/post.py --workdir runs/wing_body [--base 0] [--final N] [--out figures]
(--final defaults to final_eval from opt_result.json). The Mach-field plot needs the `vtk` Python module.
"""
import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import su2run as S  # noqa: E402
import geometry as G  # noqa: E402

RUNS, FIG, XCG = None, None, G.X_CG
GAM, RGAS, PINF, TINF, MINF = 1.4, 287.058, 8850.0, 216.65, 1.7
RHOINF = PINF / (RGAS * TINF)
VINF = MINF * np.sqrt(GAM * RGAS * TINF)
QINF = 0.5 * RHOINF * VINF**2
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 130})
C0, C1 = "#4a6fa5", "#d1495b"   # baseline / optimised


def mesh_path(d):
    p = os.path.join(d, "mesh_def.su2")
    return p if os.path.exists(p) else os.path.join(RUNS, "setup", "mesh_ffd.su2")


def read_marker_tris(mesh, marker):
    c, t = S.read_su2_marker(mesh, marker)
    return c, t


def cons_to_prim(rho, mx, my, mz, E):
    v2 = (mx**2 + my**2 + mz**2) / rho**2
    p = (GAM - 1) * (E - 0.5 * rho * v2)
    a = np.sqrt(GAM * p / rho)
    return p, np.sqrt(v2) / a


def read_surface(d):
    a = np.genfromtxt(os.path.join(d, "surface_flow.csv"), delimiter=",", names=True)
    p, M = cons_to_prim(a["Density"], a["Momentum_x"], a["Momentum_y"], a["Momentum_z"], a["Energy"])
    return a["PointID"].astype(int), np.c_[a["x"], a["y"], a["z"]], (p - PINF) / QINF


def slice_area(coords, tris, xs):
    """Cross-section area at x = const: A = contour integral of y dz (the symmetry plane gives 0);
    triangles are oriented with the normal into the flow. Returns the half-section areas and the contour segments."""
    p = coords[tris]
    nrm = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    out, segs_all = [], []
    for x in xs:
        dx = p[:, :, 0] - x
        sgn = np.sign(dx)
        cut = (sgn.min(1) < 0) & (sgn.max(1) > 0)
        A = 0.0
        segs = []
        for tri, nv in zip(p[cut], nrm[cut]):
            pts = []
            for a_, b_ in ((0, 1), (1, 2), (2, 0)):
                da, db = tri[a_, 0] - x, tri[b_, 0] - x
                if da * db < 0:
                    s = da / (da - db)
                    pts.append(tri[a_] + s * (tri[b_] - tri[a_]))
            if len(pts) != 2:
                continue
            q0, q1 = pts[0][1:], pts[1][1:]
            t = q1 - q0
            if t[1] * nv[1] - t[0] * nv[2] < 0:     # outward normal (t_z, -t_y) must agree with (n_y, n_z)
                q0, q1 = q1, q0
            A += 0.5 * (q0[0] + q1[0]) * (q1[1] - q0[1])
            segs.append((q0, q1))
        out.append(A)
        segs_all.append(segs)
    return np.array(out), segs_all


def main(d0, d1):
    res = {}
    m0, m1 = mesh_path(d0), mesh_path(d1)
    c0, t0 = S.read_su2_marker(m0, "aircraft", with_tets=True)
    c1, _ = S.read_su2_marker(m1, "aircraft")
    t1 = t0
    # ---------------- 1. side view: upper and lower fuselage lines at y = 0
    on = np.unique(t0)
    on = on[np.abs(c0[on, 1]) < 1e-6]
    fig, ax = plt.subplots(2, 1, figsize=(10, 5.6), sharex=True, gridspec_kw=dict(height_ratios=[2.2, 1]))
    up_n = on[c0[on, 2] > 1e-9]; lo_n = on[c0[on, 2] < -1e-9]     # upper/lower from the BASELINE shape
    for c, col, lab in ((c0, C0, "baseline (Sears-Haack)"), (c1, C1, "optimised")):
        for k_, nodes in enumerate((up_n, lo_n)):
            o = nodes[np.argsort(c0[nodes, 0])]
            ax[0].plot(c[o, 0], c[o, 2], color=col, lw=1.6, label=lab if k_ == 0 else None)
    # camber (mid) line and its change
    xg = np.linspace(0.5, 59.5, 240)

    def mid(c):
        up = c[up_n]; lo = c[lo_n]
        zu = np.interp(xg, *zip(*sorted(zip(up[:, 0], up[:, 2]))))
        zl = np.interp(xg, *zip(*sorted(zip(lo[:, 0], lo[:, 2]))))
        return 0.5 * (zu + zl), zu - zl
    zm0, h0 = mid(c0); zm1, h1 = mid(c1)
    ax[0].plot(xg, zm1, "--", color=C1, lw=1, label="mid line (optimised)")
    ax[0].axvline(XCG, color="k", lw=0.8, ls=":")
    ax[0].text(XCG + 0.4, -2.3, "CG", fontsize=9)
    ax[0].set_ylabel("z, m"); ax[0].set_ylim(-2.6, 2.6); ax[0].legend(loc="upper right", fontsize=8)
    ax[0].set_title("Side view in the symmetry plane: fuselage contour before and after optimisation")
    ax[1].plot(xg, zm1 - zm0, color=C1, label="mid-line shift Δz, m")
    ax[1].plot(xg, h1 - h0, color="#2a9d8f", label="section height change Δh, m")
    ax[1].axhline(0, color="k", lw=0.6); ax[1].set_xlabel("x, m"); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "shape_side.png")); plt.close(fig)
    res["max_axis_shift_m"] = float(np.abs(zm1 - zm0).max())
    res["axis_shift_nose_m"] = float((zm1 - zm0)[0])
    # ---------------- 2. cross-sections
    xs = [6, 12, 20, 28, 36, 44, 52]
    _, s0 = slice_area(c0, t0, xs); _, s1 = slice_area(c1, t1, xs)
    fig, axs = plt.subplots(1, len(xs), figsize=(13, 2.7), sharey=True)
    for k, x in enumerate(xs):
        for segs, col in ((s0[k], C0), (s1[k], C1)):
            for q0, q1 in segs:
                for sg in (1, -1):
                    axs[k].plot([sg * q0[0], sg * q1[0]], [q0[1], q1[1]], color=col, lw=1.2)
        axs[k].set_aspect("equal"); axs[k].set_xlim(-2.8, 2.8); axs[k].set_ylim(-2.2, 2.2)
        axs[k].set_title(f"x = {x} m", fontsize=9)
    axs[0].set_ylabel("z, m")
    fig.suptitle("Cross-sections (blue: baseline, red: optimised; mirrored in y)", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "shape_sections.png")); plt.close(fig)
    # ---------------- 3. cross-section area distribution (area rule)
    xa = np.linspace(0.2, 59.8, 150)
    A0, _ = slice_area(c0, t0, xa); A1, _ = slice_area(c1, t1, xa)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(xa, 2 * A0, color=C0, label="baseline configuration")
    ax.plot(xa, 2 * A1, color=C1, label="optimised")
    ax.plot(xa, np.pi * G.sears_haack_r(xa) ** 2, color="gray", ls=":", label="Sears-Haack fuselage alone")
    ax.set_xlabel("x, m"); ax.set_ylabel("S(x), m²"); ax.legend()
    ax.set_title("Cross-section areas of the configuration (full, 2 x half model)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "area_distribution.png")); plt.close(fig)
    # ---------------- 4. Cp along the upper and lower lines + Cp maps
    ids0, X0, cp0 = read_surface(d0); ids1, X1, cp1 = read_surface(d1)
    fig, ax = plt.subplots(figsize=(10, 4))
    for X, cp, col, lab in ((X0, cp0, C0, "baseline"), (X1, cp1, C1, "optimised")):
        ids_ = ids0 if lab == "baseline" else ids1
        for sgn, ls in ((1, "-"), (-1, "--")):
            sel = np.isin(ids_, up_n if sgn == 1 else lo_n)
            o = np.argsort(c0[ids_[sel], 0])
            ax.plot(X[sel][o, 0], cp[sel][o], ls, color=col, lw=1.2,
                    label=f"{lab}, {'upper' if sgn == 1 else 'lower'}")
    ax.set_ylim(0.2, -0.15); ax.set_xlabel("x, m"); ax.set_ylabel("Cp"); ax.legend(fontsize=8, ncol=2)
    ax.set_title("Pressure coefficient Cp along the upper and lower fuselage lines (y = 0)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "cp_centerline.png")); plt.close(fig)
    # Cp maps, upper and lower surface
    fig, axs = plt.subplots(2, 2, figsize=(12, 3.4), sharex=True, sharey=True)
    for col_i, (cc, X, ids, cp, lab) in enumerate(((c0, X0, ids0, cp0, "baseline"), (c1, X1, ids1, cp1, "optimised"))):
        cpn = np.full(len(cc), np.nan); cpn[ids] = cp
        p = cc[t0]
        nz = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])[:, 2]
        for row, (sel, nm) in enumerate(((nz > 0, "upper surface"), (nz < 0, "lower surface"))):
            tr = mtri.Triangulation(cc[:, 0], cc[:, 1], t0[sel])
            h = axs[row, col_i].tripcolor(tr, cpn, shading="gouraud", cmap="RdBu_r", vmin=-0.12, vmax=0.12)
            axs[row, col_i].set_aspect("equal"); axs[row, col_i].set_title(f"{lab}: {nm}", fontsize=9)
            axs[row, col_i].set_xlim(-1, 61); axs[row, col_i].set_ylim(0, 15)
    fig.colorbar(h, ax=axs, shrink=0.8, label="Cp")
    for a in axs[1]: a.set_xlabel("x, m")
    for a in axs[:, 0]: a.set_ylabel("y, m")
    fig.savefig(os.path.join(FIG, "cp_map.png"), bbox_inches="tight"); plt.close(fig)
    # ---------------- 5. Mach number in the symmetry plane (optional, needs vtk and flow.vtu)
    try:
        import vtk
        from vtk.util.numpy_support import vtk_to_numpy
    except ImportError:
        vtk = None
    if vtk is not None and all(os.path.exists(os.path.join(d, "flow.vtu")) for d in (d0, d1)):
        mach_plot(d0, d1, m0, m1, vtk, vtk_to_numpy)
    # ---------------- 6. convergence of the baseline direct and adjoint runs
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    hdr, a = S.read_history(os.path.join(d0, "history.csv"))
    it_ = a[:, hdr.index("Inner_Iter")]
    ax[0].plot(it_, a[:, hdr.index("rms[Rho]")], label="rms[ρ]")
    ax[0].plot(it_, a[:, hdr.index("rms[RhoE]")], label="rms[ρE]")
    cd = a[:, hdr.index("CD")] * 1e4
    ax2 = ax[0].twinx(); ax2.plot(it_, cd, color="k", lw=0.8)
    ax2.set_ylabel("CD, counts (1e-4)"); ax2.set_ylim(0.7 * cd[-1], 1.3 * cd[-1]); ax2.grid(False)
    ax[0].set_xlabel("iteration"); ax[0].set_ylabel("log10 residual"); ax[0].legend(loc="lower left")
    ax[0].set_title("Direct problem (baseline), fixed CL", fontsize=10)
    for tag, lab in (("cd", "adjoint of CD"), ("cmy", "adjoint of CMy")):
        f = os.path.join(d0, f"history_adj_{tag}.csv")
        if os.path.exists(f):
            hh, b = S.read_history(f)
            ax[1].plot(b[:, hh.index("Inner_Iter")], b[:, hh.index("rms[A_Rho]")], label=lab)
    ax[1].set_xlabel("iteration"); ax[1].set_ylabel("log10 rms[A_ρ]"); ax[1].legend()
    ax[1].set_title("Discrete adjoint problems", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "convergence.png")); plt.close(fig)
    res["rho_drop_orders"] = float(a[0, hdr.index("rms[Rho]")] - a[:, hdr.index("rms[Rho]")].min())
    # ---------------- 7. optimisation history
    h = np.genfromtxt(os.path.join(RUNS, "dsn_history.csv"), delimiter=",", names=True)
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    ax[0].plot(h["eval"], h["K"], "o-", ms=3); ax[0].set_title("K = CL/CD at fixed CL"); ax[0].set_xlabel("solver call")
    ax[1].plot(h["eval"], h["CMy"] * 1e3, "o-", ms=3, color=C1); ax[1].axhline(0, color="k", lw=0.6)
    ax[1].set_title("CMy about the CG, x1000"); ax[1].set_xlabel("solver call")
    ax[2].plot(h["eval"], h["Vfus_rel"] * 100, "o-", ms=3, color="#2a9d8f")
    ax[2].axhline(100 * VOL_FRAC, color="k", lw=0.6, ls="--")
    ax[2].set_title("Fuselage volume, % of baseline"); ax[2].set_xlabel("solver call")
    itf = os.path.join(RUNS, "opt_iterations.txt")
    ev_it = [int(l.split("eval")[1].split()[0]) for l in open(itf)] if os.path.exists(itf) else []
    if ev_it:
        sel = np.isin(h["eval"], ev_it)
        ax[0].plot(h["eval"][sel], h["K"][sel], "s", color="k", ms=5, label="accepted SLSQP iterations")
        ax[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "opt_history.png")); plt.close(fig)
    print(json.dumps(res, indent=1))


def mach_plot(d0, d1, m0, m1, vtk, vtk_to_numpy):
    from scipy.spatial import cKDTree
    fig, axs = plt.subplots(2, 1, figsize=(11, 6.4), sharex=True)
    for k, (d, mesh, lab) in enumerate(((d0, m0, "baseline"), (d1, m1, "optimised"))):
        rd = vtk.vtkXMLUnstructuredGridReader(); rd.SetFileName(os.path.join(d, "flow.vtu")); rd.Update()
        g = rd.GetOutput(); pts = vtk_to_numpy(g.GetPoints().GetData())
        pdta = g.GetPointData()
        get = lambda n: vtk_to_numpy(pdta.GetArray(n))  # noqa: E731
        mom = get("Momentum")
        _, M = cons_to_prim(get("Density"), mom[:, 0], mom[:, 1], mom[:, 2], get("Energy"))
        cs, ts = S.read_su2_marker(mesh, "symmetry")
        idx = cKDTree(pts).query(cs[np.unique(ts)])[1]      # node order in the .vtu may differ from the mesh
        Mn = np.full(len(cs), np.nan); Mn[np.unique(ts)] = M[idx]
        tr = mtri.Triangulation(cs[:, 0], cs[:, 2], ts)
        h = axs[k].tricontourf(tr, Mn, levels=np.linspace(1.35, 1.95, 25), cmap="viridis", extend="both")
        axs[k].set_aspect("equal"); axs[k].set_xlim(-4, 72); axs[k].set_ylim(-14, 14)
        axs[k].set_title(f"Mach number in the symmetry plane: {lab}", fontsize=10); axs[k].set_ylabel("z, m")
    axs[1].set_xlabel("x, m")
    fig.colorbar(h, ax=axs, shrink=0.85, label="M")
    fig.savefig(os.path.join(FIG, "mach_symmetry.png"), bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default="runs/wing_body")
    ap.add_argument("--base", type=int, default=0)
    ap.add_argument("--final", type=int, default=None)
    ap.add_argument("--out", default="figures")
    ap.add_argument("--xcg", type=float, default=None)
    args = ap.parse_args()
    RUNS = os.path.abspath(args.workdir)
    FIG = os.path.abspath(args.out)
    os.makedirs(FIG, exist_ok=True)
    st = os.path.join(RUNS, "setup", "settings.json")
    settings = json.load(open(st)) if os.path.exists(st) else {}
    XCG = args.xcg if args.xcg is not None else settings.get("xcg", G.X_CG)
    VOL_FRAC = settings.get("vol_frac", 0.995)
    final = args.final
    if final is None:
        final = json.load(open(os.path.join(RUNS, "opt_result.json")))["final_eval"]
    main(os.path.join(RUNS, f"dsn_{args.base:03d}"), os.path.join(RUNS, f"dsn_{final:03d}"))
