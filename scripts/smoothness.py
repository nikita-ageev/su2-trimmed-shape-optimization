"""Smoothness metrics of a body ("waves" and "warts") from SU2 surface output.

The surface triangulation of a marker (connectivity from the mesh file; coordinates and the flow from
surface_flow.csv of a design, matched by PointID) is cut by planes x = const inside a zone [x0, x1] where the
body is alone (the nose up to the wing). On every cut the radius r(phi) from the body axis z = zc(x) is taken on
five rays, phi = 0 (crest), 45, 90 (side), 135, 180 deg (keel), and Cp is interpolated at the same points.
Per ray, on an x grid with step h (about the surface mesh size):

  warts   interior local extrema of r(x) with prominence > 2e-4 L: a nose rises monotonically from the tip, so
          every such extremum is a bump or a dent
  infl    sign changes of r''(x) (2nd-order Savitzky-Golay over 5 points) where |r''| > 1/L: number of waves
  cp_ext  local extrema of Cp(x) with prominence > 0.01: pressure waves
  cp_tv   total variation of Cp(x)

For two designs on the same mesh, d(x) = r_final - r_base:
  d_ext   local extrema of d(x) with prominence > 1e-4 L: waves added by the optimiser
  d_curv  max |d''| (1/m): curvature added by the optimiser

Usage:  python scripts/smoothness.py <mesh.su2> <base dsn dir> [<final dsn dir>] [--zone x0 x1] [--h 0.25]
"""
import argparse
import json
import math
import os

import numpy as np
from scipy.signal import find_peaks, savgol_filter

GAMMA = 1.4
RAYS = (0, 45, 90, 135, 180)


def surface_tris(mesh, marker="aircraft"):
    """Triangles (global node ids) of one marker, without reading the volume elements."""
    tris = []
    with open(mesh) as f:
        for line in f:
            if line.startswith("MARKER_TAG=") and line.split("=")[1].strip() == marker:
                n = int(next(f).split("=")[1])
                for _ in range(n):
                    tris.append([int(v) for v in next(f).split()[1:4]])
                break
    return np.array(tris, dtype=np.int64).reshape(-1, 3)


def load_surface(csv, pinf, mach):
    """Node coordinates and Cp from SU2's surface_flow.csv (conservative variables)."""
    a = np.genfromtxt(csv, delimiter=",", skip_header=1, ndmin=2)
    pid = a[:, 0].astype(np.int64)
    y = np.where(np.abs(a[:, 2]) < 1e-7, 0.0, a[:, 2])          # symmetry-plane nodes carry ~1e-17 noise
    rho, mx, my, mz, E = a[:, 4], a[:, 5], a[:, 6], a[:, 7], a[:, 8]
    p = (GAMMA - 1.0) * (E - 0.5 * (mx ** 2 + my ** 2 + mz ** 2) / rho)
    lut = np.full(pid.max() + 1, -1, dtype=np.int64)
    lut[pid] = np.arange(len(pid))
    return dict(X=np.c_[a[:, 1], y, a[:, 3]], cp=(p - pinf) / (0.5 * GAMMA * pinf * mach ** 2), lut=lut)


def cut(X, cp, tris, x, zc, rays=RAYS):
    """Radius and Cp on the rays of the plane cut x = const (outermost crossing per ray)."""
    P = X[tris]
    s = P[:, :, 0] - x
    m = (s.min(axis=1) < 0) & (s.max(axis=1) > 0)
    P, s, C = P[m], s[m], cp[tris[m]]
    ends = [[] for _ in range(len(P))]
    for a, b in ((0, 1), (1, 2), (2, 0)):
        e = np.where(s[:, a] * s[:, b] < 0)[0]
        t = s[e, a] / (s[e, a] - s[e, b])
        q = P[e, a] + t[:, None] * (P[e, b] - P[e, a])
        c = C[e, a] + t * (C[e, b] - C[e, a])
        for i, qq, cc in zip(e, q, c):
            ends[i].append((qq[1], qq[2] - zc, cc))
    seg = [v for v in ends if len(v) == 2]
    if not seg:
        return [np.nan] * len(rays), [np.nan] * len(rays)
    A = np.array([v[0] for v in seg]); B = np.array([v[1] for v in seg])
    rr, cc = [], []
    for ph in rays:
        d = np.array([math.sin(math.radians(ph)), math.cos(math.radians(ph))])     # (y, z): 0 = up, 180 = down
        n = np.array([-d[1], d[0]])
        sa, sb = A[:, :2] @ n, B[:, :2] @ n
        hit = (np.minimum(sa, sb) <= 1e-9) & (np.maximum(sa, sb) >= -1e-9)
        if not hit.any():
            rr.append(np.nan); cc.append(np.nan); continue
        den = sa[hit] - sb[hit]
        t = np.clip(np.where(np.abs(den) > 1e-12, sa[hit] / np.where(np.abs(den) > 1e-12, den, 1.0), 0.5), 0.0, 1.0)
        Q = A[hit, :2] + t[:, None] * (B[hit, :2] - A[hit, :2])
        r = Q @ d
        c = A[hit, 2] + t * (B[hit, 2] - A[hit, 2])
        if not (r > 0).any():
            rr.append(np.nan); cc.append(np.nan); continue
        k = int(np.argmax(np.where(r > 0, r, -np.inf)))
        rr.append(float(r[k])); cc.append(float(c[k]))
    return rr, cc


def profiles(mesh, dsn_dir, zone, h, pinf, mach, zc=lambda x: 0.0, marker="aircraft"):
    """x grid and r(x, ray), Cp(x, ray) of one design."""
    s = load_surface(os.path.join(dsn_dir, "surface_flow.csv"), pinf, mach)
    T = s["lut"][surface_tris(mesh, marker)]
    T = T[(T >= 0).all(axis=1)]
    xs = np.arange(zone[0], zone[1], h)
    R, CP = [], []
    for x in xs:
        r, c = cut(s["X"], s["cp"], T, x, float(zc(x)))
        R.append(r); CP.append(c)
    return xs, np.array(R), np.array(CP)


def n_extrema(v, tol):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) < 3:
        return 0, 0.0
    a, pa = find_peaks(v, prominence=tol)
    b, pb = find_peaks(-v, prominence=tol)
    pr = np.concatenate([pa["prominences"], pb["prominences"]])
    return int(len(a) + len(b)), float(pr.max() if len(pr) else 0.0)


def second_derivative(xs, v):
    v = np.asarray(v, float)
    ok = np.isfinite(v)
    if ok.sum() < 7:
        return np.zeros(0)
    return savgol_filter(np.interp(xs, xs[ok], v[ok]), 5, 2, deriv=2, delta=xs[1] - xs[0])


def sign_changes(v, tol):
    s = np.sign(v[np.abs(v) > tol])
    return int(np.sum(s[1:] != s[:-1]))


def ray_metrics(xs, r, cp, L):
    k = second_derivative(xs, r)
    nw, pw = n_extrema(r, 2e-4 * L)
    return dict(warts=nw, wart_max_m=pw, infl=sign_changes(k, 1.0 / L),
                cp_ext=n_extrema(cp, 0.01)[0], cp_tv=float(np.nansum(np.abs(np.diff(cp)))))


def evaluate(mesh, base_dir, final_dir=None, zone=(0.6, 24.0), h=0.25, L=60.0, pinf=8850.0, mach=1.7,
             zc=lambda x: 0.0):
    xs, Rb, Cb = profiles(mesh, base_dir, zone, h, pinf, mach, zc)
    out = dict(zone=list(zone), h=h, rays=list(RAYS),
               base={f"r{ph}": ray_metrics(xs, Rb[:, i], Cb[:, i], L) for i, ph in enumerate(RAYS)})
    if final_dir:
        _, Rf, Cf = profiles(mesh, final_dir, zone, h, pinf, mach, zc)
        out["final"] = {f"r{ph}": ray_metrics(xs, Rf[:, i], Cf[:, i], L) for i, ph in enumerate(RAYS)}
        ch = {}
        for i, ph in enumerate(RAYS):
            d = Rf[:, i] - Rb[:, i]
            ne, pe = n_extrema(d, 1e-4 * L)
            k = second_derivative(xs, d)
            ch[f"r{ph}"] = dict(d_max_m=float(np.nanmax(np.abs(d))) if np.isfinite(d).any() else 0.0, d_ext=ne,
                                d_ext_prom_m=pe, d_curv=float(np.abs(k).max()) if len(k) else 0.0)
        out["change"] = ch
    return out


def totals(m, key):
    """Sums over the five rays (max for amplitudes)."""
    b = m[key]
    t = dict(warts=sum(v["warts"] for v in b.values()), infl=sum(v["infl"] for v in b.values()),
             cp_ext=sum(v["cp_ext"] for v in b.values()), cp_tv=round(sum(v["cp_tv"] for v in b.values()), 3),
             wart_max_mm=round(1e3 * max(v["wart_max_m"] for v in b.values()), 1))
    if key == "final" and "change" in m:
        c = m["change"]
        t.update(d_ext=sum(v["d_ext"] for v in c.values()), d_curv=round(max(v["d_curv"] for v in c.values()), 4),
                 d_max_mm=round(1e3 * max(v["d_max_m"] for v in c.values()), 1))
    return t


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mesh")
    ap.add_argument("base")
    ap.add_argument("final", nargs="?")
    ap.add_argument("--zone", type=float, nargs=2, default=(0.6, 23.4), help="x range, m (default: nose to wing root)")
    ap.add_argument("--h", type=float, default=0.25)
    ap.add_argument("--length", type=float, default=60.0)
    ap.add_argument("--pinf", type=float, default=8850.0)
    ap.add_argument("--mach", type=float, default=1.7)
    a = ap.parse_args()
    m = evaluate(a.mesh, a.base, a.final, tuple(a.zone), a.h, a.length, a.pinf, a.mach)
    print(json.dumps(m, indent=1))
    print("base ", totals(m, "base"))
    if a.final:
        print("final", totals(m, "final"))


if __name__ == "__main__":
    main()
