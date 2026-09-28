"""Thin wrapper around the SU2 executables.

* config templating (``{PLACEHOLDER}`` substitution),
* running SU2_DEF / SU2_CFD / SU2_CFD_AD / SU2_DOT_AD (optionally under MPI),
* reading SU2 history files and gradient files,
* body volume from the surface mesh and its analytic gradient,
* a Python re-implementation of SU2's FFD map (Bernstein or uniform B-spline blending)
  that reproduces SU2_DEF to ~1e-10 and is used for the volume constraint.

Environment variables:
  SU2_RUN   directory with SU2 binaries (the standard SU2 variable); otherwise binaries are taken from PATH
  SU2_NP    number of MPI ranks (default 4); 1 runs without mpirun
  MPIRUN    MPI launcher (default "mpirun")
"""
import csv
import os
import re
import shutil
import subprocess
import time
from math import comb

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NP = int(os.environ.get("SU2_NP", "4"))
MPIRUN = os.environ.get("MPIRUN", "mpirun")
TIMING_CSV = None          # set by the drivers: every SU2 call is appended here

ENV = dict(os.environ)
ENV.setdefault("OMPI_MCA_rmaps_base_oversubscribe", "1")


def su2_exe(name):
    d = os.environ.get("SU2_RUN")
    if d and os.path.exists(os.path.join(d, name)):
        return os.path.join(d, name)
    p = shutil.which(name)
    if p:
        return p
    raise FileNotFoundError(f"{name} not found: set SU2_RUN to the SU2 bin directory or add it to PATH")


def check_su2():
    """Fail early if the AD-enabled binaries are missing."""
    for exe in ("SU2_CFD", "SU2_DEF", "SU2_CFD_AD", "SU2_DOT_AD"):
        su2_exe(exe)


# ------------------------------------------------------------------ config and runs

def fill_template(template_path, fields, out_path):
    txt = open(template_path).read()
    for k, v in fields.items():
        txt = txt.replace("{" + k + "}", str(v))
    left = re.findall(r"\{[A-Z_0-9]+\}", re.sub(r"%.*", "", txt))
    assert not left, f"unfilled placeholders in {out_path}: {sorted(set(left))}"
    with open(out_path, "w") as f:
        f.write(txt)


def run(exe, cfg, cwd, log, np_=None):
    """Run an SU2 executable in ``cwd`` with a config file name relative to ``cwd``.
    SU2 config files must not contain paths with spaces, so everything is kept relative."""
    np_ = NP if np_ is None else np_
    t0 = time.time()
    path = su2_exe(exe)
    cmd = [MPIRUN, "-n", str(np_), path, cfg] if np_ > 1 else [path, cfg]
    with open(os.path.join(cwd, log), "w") as f:
        r = subprocess.run(cmd, cwd=cwd, env=ENV, stdout=f, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    if TIMING_CSV:
        with open(TIMING_CSV, "a") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')},{os.path.basename(cwd)},{exe},{dt:.1f},{r.returncode}\n")
    if r.returncode != 0:
        raise RuntimeError(f"{exe} failed in {cwd}, see {os.path.join(cwd, log)}")
    return dt


def read_history(path):
    rows = list(csv.reader(open(path)))
    hdr = [h.strip().strip('"') for h in rows[0]]
    data = [[float(v) for v in r] for r in rows[1:] if r]
    return hdr, np.array(data)


def read_meta(path):
    """flow.meta written by SU2 next to the restart (AoA, DCD_DCL_VALUE, DCMY_DCL_VALUE, ...)."""
    out = {}
    for line in open(path):
        if "= " in line:
            k, v = line.strip().split("= ", 1)
            out[k.strip()] = v.strip()
    return out


def direct_summary(cwd):
    hdr, a = read_history(os.path.join(cwd, "history.csv"))
    last = dict(zip(hdr, a[-1]))
    meta = read_meta(os.path.join(cwd, "flow.meta"))
    return dict(CL=last["CL"], CD=last["CD"], CMy=last["CMy"],
                AoA=float(meta.get("AOA", last.get("AoA", 0.0))),
                dCD_dCL=float(meta.get("DCD_DCL_VALUE", 0.0)), dCMy_dCL=float(meta.get("DCMY_DCL_VALUE", 0.0)),
                rho0=a[0, hdr.index("rms[Rho]")], rho_end=a[-1, hdr.index("rms[Rho]")], iters=len(a))


def read_grad(path):
    """of_grad*.csv from SU2_DOT_AD: last column of every row after the header."""
    lines = [l.strip() for l in open(path) if l.strip()]
    return np.array([float(l.split(",")[-1]) for l in lines[1:]])


# ------------------------------------------------------------------ FFD design variables

class FFDSpec:
    """Rectangular FFD box and the list of SU2 design variables (FFD_CONTROL_POINT).

    box: dict(x=(x0, x1), y=(y0, y1), z=(z0, z1), deg=(L, M, N), blend=None | (ox, oy, oz), fixj=J, def_nl=n)
    Design variables: z-displacements of all control points with j < J,
                      y-displacements of control points with 1 <= j < J (points on the symmetry plane keep y = 0).
    x-displacements are never used, so the length of the body is fixed.
    """

    def __init__(self, name, box):
        self.name, self.box = name, box
        L, M, N = box["deg"]
        J = box["fixj"]
        self.dvs = [(i, j, k, 2) for i in range(L + 1) for j in range(J) for k in range(N + 1)]
        self.dvs += [(i, j, k, 1) for i in range(L + 1) for j in range(1, J) for k in range(N + 1)]
        self.ndv = len(self.dvs)

    def dv_strings(self, x, tag="FUS"):
        kind = ", ".join(["FFD_CONTROL_POINT"] * self.ndv)
        params = "; ".join(f"( {tag}, {i}, {j}, {k}, {1.0 if d == 0 else 0.0}, {1.0 if d == 1 else 0.0}, "
                           f"{1.0 if d == 2 else 0.0} )" for i, j, k, d in self.dvs)
        vals = ", ".join(f"{v:.10g}" for v in x)
        return dict(DV_KIND=kind, DV_PARAM=params, DV_VALUE=vals)

    def cfg_fields(self, tag="FUS"):
        b = self.box
        x0, x1 = b["x"]; y0, y1 = b["y"]; z0, z1 = b["z"]
        blend = ("FFD_BLENDING= BSPLINE_UNIFORM\nFFD_BSPLINE_ORDER= (%d, %d, %d)" % b["blend"]) if b["blend"] \
            else "FFD_BLENDING= BEZIER"
        return dict(FFD_DEF=f"({tag}, {x0},{y0},{z0}, {x1},{y0},{z0}, {x1},{y1},{z0}, {x0},{y1},{z0}, "
                            f"{x0},{y0},{z1}, {x1},{y0},{z1}, {x1},{y1},{z1}, {x0},{y1},{z1})",
                    FFD_DEG="(%d, %d, %d)" % b["deg"], FFD_FIXJ=str(b["fixj"]), DEF_NL=str(b["def_nl"]),
                    FFD_BLEND=blend)


def bernstein(n, i, t):
    return comb(n, i) * t**i * (1 - t) ** (n - i)


def bspline_basis(n_ctrl, order, t):
    """Open uniform B-spline basis as in SU2 (CBSplineBlending): matrix len(t) x n_ctrl."""
    from scipy.interpolate import BSpline
    U = [0.0] * order + [(k + 1) / (n_ctrl - order + 1) for k in range(n_ctrl - order)] + [1.0] * order
    t = np.clip(np.asarray(t, float), 0.0, 1.0)
    return BSpline(np.array(U), np.eye(n_ctrl), order - 1, extrapolate=True)(t)


def basis_1d(box, deg, axis, t):
    if box["blend"] is None:
        return np.array([bernstein(deg, i, t) for i in range(deg + 1)]).T
    return bspline_basis(deg + 1, box["blend"][axis], t)


class FFD:
    """FFD map of a rectangular box, same formula as SU2: X = X0 + sum_idv B_idv(u, v, w) * x_idv * e_dir."""

    def __init__(self, coords, spec):
        bx = spec.box
        self.spec = spec
        self.inside = ((coords[:, 0] >= bx["x"][0]) & (coords[:, 0] <= bx["x"][1]) &
                       (coords[:, 1] >= bx["y"][0] - 1e-9) & (coords[:, 1] <= bx["y"][1]) &
                       (coords[:, 2] >= bx["z"][0]) & (coords[:, 2] <= bx["z"][1]))
        u = (coords[:, 0] - bx["x"][0]) / (bx["x"][1] - bx["x"][0])
        v = (coords[:, 1] - bx["y"][0]) / (bx["y"][1] - bx["y"][0])
        w = (coords[:, 2] - bx["z"][0]) / (bx["z"][1] - bx["z"][0])
        L, M, N = bx["deg"]
        if bx["blend"] is not None:
            # with B-splines and equally spaced control points the map (u,v,w) -> (x,y,z) is not the identity
            # (SU2 inverts it by Newton iterations); here it is inverted per axis with a dense table
            u, v, w = self._invert(L, 0, u), self._invert(M, 1, v), self._invert(N, 2, w)
        bu, bv, bw = basis_1d(bx, L, 0, u), basis_1d(bx, M, 1, v), basis_1d(bx, N, 2, w)
        self.B = np.zeros((len(coords), spec.ndv))
        for idv, (i, j, k, d) in enumerate(spec.dvs):
            self.B[:, idv] = bu[:, i] * bv[:, j] * bw[:, k] * self.inside
        self.dirs = np.array([d for (_, _, _, d) in spec.dvs])
        self.X0 = coords.copy()

    def _invert(self, deg, axis, s):
        tt = np.linspace(0.0, 1.0, 200001)
        phys = basis_1d(self.spec.box, deg, axis, tt) @ (np.arange(deg + 1) / deg)
        phys[-1] = 1.0
        return np.interp(np.clip(s, 0, 1), phys, tt)

    def deform(self, x):
        X = self.X0.copy()
        for d in (1, 2):
            m = self.dirs == d
            X[:, d] += self.B[:, m] @ x[m]
        return X


# ------------------------------------------------------------------ surface mesh and volume

def read_su2_marker(mesh, marker="aircraft", with_tets=False):
    """Node coordinates and triangles of a boundary marker (global node numbering) from an .su2 file."""
    with open(mesh) as f:
        lines = f.readlines()
    i, coords, tris, tets = 0, None, None, None
    while i < len(lines):
        s = lines[i].strip()
        if s.startswith("NELEM="):
            nelem = int(s.split("=")[1])
            if with_tets:
                tets = np.array([[int(v) for v in lines[i + 1 + e].split()[1:5]] for e in range(nelem)
                                 if lines[i + 1 + e].split()[0] == "10"])
            i += nelem + 1
            continue
        if s.startswith("NPOIN="):
            n = int(s.split("=")[1].split()[0])
            coords = np.array([[float(v) for v in lines[i + 1 + p].split()[:3]] for p in range(n)])
            i += n + 1
            continue
        if s.startswith("MARKER_TAG=") and s.split("=")[1].strip() == marker:
            ne = int(lines[i + 1].split("=")[1])
            tris = np.array([[int(v) for v in lines[i + 2 + e].split()[1:4]] for e in range(ne)])
            i += ne + 2
            continue
        if s.startswith("FFD_NBOX"):
            break
        i += 1
    if with_tets:
        tris = orient_outward(coords, tris, tets)
    return coords, tris


def orient_outward(coords, tris, tets):
    """Gmsh does not orient boundary triangles consistently. Orient every triangle so that its normal
    points into the flow: find the tetrahedron that owns the face and use its fourth (fluid-side) vertex."""
    key = {tuple(sorted(t)): n for n, t in enumerate(tris)}
    opp = -np.ones(len(tris), dtype=int)
    for a, b, c, d in ((0, 1, 2, 3), (0, 1, 3, 2), (0, 2, 3, 1), (1, 2, 3, 0)):
        f = np.sort(tets[:, [a, b, c]], axis=1)
        for r in range(len(f)):
            n = key.get((f[r, 0], f[r, 1], f[r, 2]))
            if n is not None:
                opp[n] = tets[r, d]
    assert (opp >= 0).all(), "no owner tetrahedron for some surface triangles"
    p0, p1, p2 = coords[tris[:, 0]], coords[tris[:, 1]], coords[tris[:, 2]]
    nrm = np.cross(p1 - p0, p2 - p0)
    flip = np.einsum("ij,ij->i", nrm, coords[opp] - p0) < 0
    tris = tris.copy()
    tris[flip] = tris[flip][:, [0, 2, 1]]
    return tris


def half_volume(coords, tris):
    """Volume of the half body: V = surface integral of y * n_y dA over the body surface
    (the symmetry plane y = 0 contributes nothing). The sign is fixed in ``Volume``."""
    p0, p1, p2 = coords[tris[:, 0]], coords[tris[:, 1]], coords[tris[:, 2]]
    nA = 0.5 * np.cross(p1 - p0, p2 - p0)
    yc = (p0[:, 1] + p1[:, 1] + p2[:, 1]) / 3.0
    return float(np.sum(yc * nA[:, 1]))


class Volume:
    """Enclosed (half) volume of the surface marker as a function of the FFD design variables,
    with an analytic gradient (node derivatives of the surface integral + chain rule through the FFD map)."""

    def __init__(self, mesh, spec, marker="aircraft"):
        c, t = read_su2_marker(mesh, marker, with_tets=True)
        self.tris = t
        self.ffd = FFD(c, spec)
        self.ndv = spec.ndv
        self.V0 = half_volume(c, t)
        self.sign = 1.0 if self.V0 > 0 else -1.0
        self.V0 *= self.sign

    def value(self, x):
        return self.sign * half_volume(self.ffd.deform(np.asarray(x)), self.tris)

    def grad(self, x):
        X = self.ffd.deform(np.asarray(x))
        t = self.tris
        p = [X[t[:, 0]], X[t[:, 1]], X[t[:, 2]]]
        dVdX = np.zeros_like(X)
        # V_tri = yc * nA_y,  nA_y = 0.5 * (a_z b_x - a_x b_z),  a = p1 - p0,  b = p2 - p0
        a, b = p[1] - p[0], p[2] - p[0]
        nAy = 0.5 * (a[:, 2] * b[:, 0] - a[:, 0] * b[:, 2])
        yc = (p[0][:, 1] + p[1][:, 1] + p[2][:, 1]) / 3.0
        for m in range(3):
            np.add.at(dVdX[:, 1], t[:, m], nAy / 3.0)
        np.add.at(dVdX[:, 2], t[:, 1], yc * 0.5 * b[:, 0])
        np.add.at(dVdX[:, 2], t[:, 2], -yc * 0.5 * a[:, 0])
        np.add.at(dVdX[:, 2], t[:, 0], yc * 0.5 * (a[:, 0] - b[:, 0]))
        g = np.zeros(self.ndv)
        for idv, d in enumerate(self.ffd.dirs):
            g[idv] = self.ffd.B[:, idv] @ dVdX[:, d]
        return self.sign * g
