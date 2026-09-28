"""Parametric wing-body geometry with CadQuery -> STEP -> Gmsh tetrahedral mesh for driver.py.

An alternative front end to make_mesh.py, which builds the same shape directly in Gmsh's OpenCASCADE kernel.
Here the fuselage is a body of revolution with the radius law

    r(x) = R (4 xi (1 - xi))^n,   xi = x / L,   0 <= xi <= xb

defined by its length L, its volume V (R follows from V), the exponent n (0.75 = Sears-Haack, the baseline
of this repository) and an optional flat base cut at xi = xb < 1. The wing is the fixed trapezoidal wing of
geometry.py. CadQuery writes a STEP file; Gmsh reads it back, cuts the flow domain and builds the same
unstructured tetrahedral mesh (markers aircraft, symmetry, farfield) that make_mesh.py produces, so the
result goes straight into driver.py / dakota_driver.py (mesh -> SU2 -> discrete adjoint -> optimiser).

Defaults reproduce the baseline: L = 60 m, V = 455.3 m^3 (R = 1.6 m), n = 0.75, no base cut.

Usage (from the repository root):
    python scripts/geometry_cadquery.py --step mesh/aircraft.step --mesh mesh/wing_body_cq.su2 --h-body 0.25
    python scripts/geometry_cadquery.py --step mesh/body.step --exponent 0.6 --base 0.95 --no-wing

Requires cadquery (pip install cadquery) for the STEP file and gmsh with its Python API for the mesh; both are
imported lazily, so the volume and radius functions work without them.
"""
import argparse
import math
import os
import sys

import numpy as np
from scipy.integrate import quad

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import geometry as G  # noqa: E402


def radius_law(xi, n=0.75):
    """Normalised radius (4 xi (1 - xi))^n; n = 0.75 is the Sears-Haack body."""
    xi = np.clip(np.asarray(xi, dtype=float), 0.0, 1.0)
    return (4.0 * xi * (1.0 - xi)) ** n


def body_volume(length, r_max, n=0.75, base=1.0):
    """Volume of the body of revolution r = r_max (4 xi (1 - xi))^n on 0 <= xi <= base (quadrature)."""
    integral = quad(lambda t: float(radius_law(t, n)) ** 2, 0.0, base, limit=200)[0]
    return math.pi * r_max ** 2 * length * integral


def r_max_for_volume(length, volume, n=0.75, base=1.0):
    """Maximum radius that gives the requested volume."""
    return math.sqrt(volume / body_volume(length, 1.0, n, base))


def meridian(length, r_max, n=0.75, base=1.0, npts=61):
    """(x, r) points of the profile, cosine-clustered at both ends; the last point is on the base plane."""
    xi = base * (0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, npts)))
    return [(float(length * t), float(r_max * radius_law(t, n))) for t in xi]


SEAM_DEG = 45.0     # angle of the seam plane of the two half revolutions, away from z = 0 (wing) and y = 0 (symmetry)


def fuselage(length=G.L_FUS, volume=G.V_FUS_FULL, n=0.75, base=1.0):
    """CadQuery solid of the body of revolution (axis x, nose at the origin).

    Built from two 180-degree revolutions instead of one 360-degree face: a full face of revolution is
    periodic, and after the boolean union with the wing its trimming does not survive the STEP export
    (OpenCASCADE writes only the patch inside the wing-body intersection loop). Two faces with their seams
    in a plane at SEAM_DEG round-trip through STEP correctly."""
    import cadquery as cq
    r_max = r_max_for_volume(length, volume, n, base)
    pts = meridian(length, r_max, n, base)
    c, s = math.cos(math.radians(SEAM_DEG)), math.sin(math.radians(SEAM_DEG))
    normal = (0.0, -s, c)                                     # plane through the x axis, rotated by SEAM_DEG
    wp = cq.Workplane(cq.Plane(origin=(0, 0, 0), xDir=(1, 0, 0), normal=normal))
    wp = wp.moveTo(*pts[0]).spline(pts[1:], includeCurrent=True)
    if base < 1.0:
        wp = wp.lineTo(pts[-1][0], 0.0)          # flat base
    half = wp.close().revolve(180.0, (0, 0, 0), (1, 0, 0))
    return half.union(half.mirror(normal, (0, 0, 0)))


def wing():
    """CadQuery solid of the wing of geometry.py: ruled loft between the root section (slightly beyond the
    symmetry plane, so that the fuse with the fuselage is clean) and the tip section, biconvex profile."""
    import cadquery as cq

    def section(x_le, y, c, npts=25):
        xi = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, npts))
        zt = 2.0 * G.T_C * c * xi * (1.0 - xi)
        up = [cq.Vector(x_le + c * s, y, G.Z_WING + z) for s, z in zip(xi, zt)]
        lo = [cq.Vector(x_le + c * s, y, G.Z_WING - z) for s, z in zip(xi[::-1], zt[::-1])]
        return cq.Wire.assembleEdges([cq.Edge.makeSpline(up), cq.Edge.makeSpline(lo)])

    w_root = section(G.X_LE_ROOT - 0.3 * math.tan(math.radians(G.SWEEP_LE_DEG)), -0.3, G.C_ROOT)
    w_tip = section(G.X_LE_TIP, G.SEMI_SPAN, G.C_TIP)
    return cq.Solid.makeLoft([w_root, w_tip], ruled=True)


def aircraft(length=G.L_FUS, volume=G.V_FUS_FULL, n=0.75, base=1.0, with_wing=True):
    """Fuselage fused with the wing (a cadquery Workplane holding one solid)."""
    body = fuselage(length, volume, n, base)
    return body.union(wing()) if with_wing else body


def write_step(shape, path):
    import cadquery as cq
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    cq.exporters.export(shape, path)
    return path


def mesh_from_step(step, out, h_body=0.25, h_far=4.0, threads=4, length=G.L_FUS):
    """Import the STEP file into Gmsh and build the mesh of make_mesh.mesh_domain around it."""
    import gmsh
    import make_mesh as MM
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add("cadquery")
    try:
        shapes = gmsh.model.occ.importShapes(step)
        solids = [s for s in shapes if s[0] == 3]
        assert solids, f"no solid in {step}"
        gmsh.model.occ.synchronize()
        x0, _, _, x1, _, _ = gmsh.model.getBoundingBox(3, solids[0][1])
        # STEP files carry a unit; the coordinates must come back unscaled (metres as written)
        assert abs((x1 - x0) - length) < 1e-3 * length, f"imported length {x1 - x0:.3f} != {length}"
        return MM.mesh_domain(solids, out, h_body, h_far, threads)
    finally:
        gmsh.finalize()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step", default="mesh/aircraft.step", help="STEP file to write")
    ap.add_argument("--mesh", default=None, help="SU2 mesh to write (omit to stop after the STEP file)")
    ap.add_argument("--length", type=float, default=G.L_FUS, help="fuselage length, m")
    ap.add_argument("--volume", type=float, default=G.V_FUS_FULL, help="fuselage volume (full body), m^3")
    ap.add_argument("--exponent", type=float, default=0.75, help="radius-law exponent n (0.75 = Sears-Haack)")
    ap.add_argument("--base", type=float, default=1.0, help="base cut at x = base * length (1 = pointed tail)")
    ap.add_argument("--no-wing", action="store_true", help="fuselage only")
    ap.add_argument("--h-body", type=float, default=0.25)
    ap.add_argument("--h-far", type=float, default=4.0)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args(argv)
    r_max = r_max_for_volume(a.length, a.volume, a.exponent, a.base)
    shape = aircraft(a.length, a.volume, a.exponent, a.base, with_wing=not a.no_wing)
    v_cad = shape.val().Volume()
    print(f"R_max = {r_max:.4f} m, L/D = {a.length / (2 * r_max):.2f}, analytic body volume "
          f"{body_volume(a.length, r_max, a.exponent, a.base):.3f} m^3, CAD solid volume {v_cad:.3f} m^3")
    write_step(shape, a.step)
    print(f"wrote {a.step} ({os.path.getsize(a.step) / 1024:.0f} kB)")
    if a.mesh:
        mesh_from_step(a.step, a.mesh, a.h_body, a.h_far, a.threads, a.length)


if __name__ == "__main__":
    main()
