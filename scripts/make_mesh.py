"""Gmsh mesh for the half model "Sears-Haack fuselage + thin wing".

Usage: python scripts/make_mesh.py <out.su2> [h_body] [h_far] [threads]
Unstructured tetrahedral mesh; markers: aircraft, symmetry, farfield.
h_body = 0.25 m gives ~350k tetrahedra (the published run), 0.6 m gives a quick ~90k-cell smoke-test mesh.

The geometry is built directly in Gmsh's OpenCASCADE kernel (build_aircraft). geometry_cadquery.py builds a
parametric version of the same shape with CadQuery, writes a STEP file and calls mesh_domain() below on it.
"""
import os
import sys

import numpy as np
import gmsh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import geometry as G  # noqa: E402

# flow domain: box in the half space y >= 0; at supersonic speed disturbances do not travel upstream
BX0, BX1, BY1, BZ = -6.0, 72.0, 32.0, 32.0


def build_aircraft():
    """Fuselage (body of revolution with the Sears-Haack radius) fused with the wing (ruled loft between a
    root and a tip biconvex section) in gmsh.model.occ; returns the dim-tags of the resulting solid(s)."""
    occ = gmsh.model.occ
    xs = G.L_FUS * (0.5 - 0.5 * np.cos(np.linspace(0, np.pi, 61)))   # clustered at the ends
    pts = [occ.addPoint(x, 0, float(G.sears_haack_r(x))) for x in xs]
    prof = occ.addSpline(pts)
    axis = occ.addLine(pts[-1], pts[0])
    loop = occ.addCurveLoop([prof, axis])
    face = occ.addPlaneSurface([loop])
    fus = occ.revolve([(2, face)], 0, 0, 0, 1, 0, 0, 2 * np.pi)
    fus_vol = [d for d in fus if d[0] == 3]

    def section(x_le, y, c):
        n = 25
        xi = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, n))
        zt = 2 * G.T_C * c * xi * (1 - xi)
        p_le = occ.addPoint(x_le, y, G.Z_WING)
        p_te = occ.addPoint(x_le + c, y, G.Z_WING)
        up = [p_le] + [occ.addPoint(x_le + c * s, y, G.Z_WING + z) for s, z in zip(xi[1:-1], zt[1:-1])] + [p_te]
        lo = [p_te] + [occ.addPoint(x_le + c * s, y, G.Z_WING - z)
                       for s, z in zip(xi[1:-1][::-1], zt[1:-1][::-1])] + [p_le]
        return occ.addWire([occ.addSpline(up), occ.addSpline(lo)])

    # root section slightly beyond the symmetry plane, tip section
    w_root = section(G.X_LE_ROOT - 0.3 * np.tan(np.radians(G.SWEEP_LE_DEG)), -0.3, G.C_ROOT)
    w_tip = section(G.X_LE_TIP, G.SEMI_SPAN, G.C_TIP)
    wing = occ.addThruSections([w_root, w_tip], makeSolid=True, makeRuled=True)
    wing_vol = [d for d in wing if d[0] == 3]
    aircraft, _ = occ.fuse(fus_vol, wing_vol)
    return aircraft


def mesh_domain(aircraft, out, h_body=0.25, h_far=4.0, threads=4):
    """Cut the flow box y >= 0 by the aircraft solid(s) (OCC dim-tags), tag the markers, set the size fields
    (fine at the body, finer at the wing edges, growing towards the far field) and write the tetrahedral mesh
    to `out` (.su2). Returns (tetrahedra, nodes)."""
    occ = gmsh.model.occ
    h_edge = 0.45 * h_body
    box = occ.addBox(BX0, 0.0, -BZ, BX1 - BX0, BY1, 2 * BZ)
    occ.cut([(3, box)], aircraft)
    occ.synchronize()

    vols = gmsh.model.getEntities(3)
    assert len(vols) == 1, vols
    surfs = gmsh.model.getBoundary(vols, oriented=False)
    sym, far, body = [], [], []
    tol = 1e-3
    for d, s in surfs:
        x0, y0, z0, x1, y1, z1 = gmsh.model.getBoundingBox(2, s)
        if abs(y0) < tol and abs(y1) < tol:
            sym.append(s)
        elif (abs(x0 - BX0) < tol and abs(x1 - BX0) < tol) or (abs(x0 - BX1) < tol and abs(x1 - BX1) < tol) \
                or (abs(y0 - BY1) < tol and abs(y1 - BY1) < tol) or abs(z0 + BZ) < tol and abs(z1 + BZ) < tol \
                or (abs(z0 - BZ) < tol and abs(z1 - BZ) < tol):
            far.append(s)
        else:
            body.append(s)
    print("symmetry", sym, "farfield", far, "aircraft", body)
    assert body and sym and far, "marker detection failed"
    gmsh.model.addPhysicalGroup(2, body, name="aircraft")
    gmsh.model.addPhysicalGroup(2, sym, name="symmetry")
    gmsh.model.addPhysicalGroup(2, far, name="farfield")
    gmsh.model.addPhysicalGroup(3, [vols[0][1]], name="fluid")

    body_curves = set()
    for s in body:
        for _, c in gmsh.model.getBoundary([(2, s)], oriented=False):
            body_curves.add(abs(c))
    f1 = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(f1, "SurfacesList", body)
    gmsh.model.mesh.field.setNumber(f1, "Sampling", 200)
    t1 = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(t1, "InField", f1)
    gmsh.model.mesh.field.setNumber(t1, "SizeMin", h_body)
    gmsh.model.mesh.field.setNumber(t1, "SizeMax", h_far)
    gmsh.model.mesh.field.setNumber(t1, "DistMin", 0.3)
    gmsh.model.mesh.field.setNumber(t1, "DistMax", 28.0)
    f2 = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(f2, "CurvesList", sorted(body_curves))
    gmsh.model.mesh.field.setNumber(f2, "Sampling", 400)
    t2 = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(t2, "InField", f2)
    gmsh.model.mesh.field.setNumber(t2, "SizeMin", h_edge)
    gmsh.model.mesh.field.setNumber(t2, "SizeMax", h_far)
    gmsh.model.mesh.field.setNumber(t2, "DistMin", 0.1)
    gmsh.model.mesh.field.setNumber(t2, "DistMax", 12.0)
    fm = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(fm, "FieldsList", [t1, t2])
    gmsh.model.mesh.field.setAsBackgroundMesh(fm)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.option.setNumber("Mesh.Algorithm3D", 10)   # HXT, parallel
    gmsh.option.setNumber("General.NumThreads", threads)
    gmsh.option.setNumber("Mesh.Optimize", 1)
    gmsh.model.mesh.generate(3)
    ntet = sum(len(t) for t in gmsh.model.mesh.getElementsByType(4)[:1])
    nodes = len(gmsh.model.mesh.getNodes()[0])
    print(f"TETS {ntet}  NODES {nodes}")
    if os.path.dirname(out):
        os.makedirs(os.path.dirname(out), exist_ok=True)
    gmsh.write(out)
    return ntet, nodes


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    out = argv[0] if len(argv) > 0 else "mesh/wing_body.su2"
    h_body = float(argv[1]) if len(argv) > 1 else 0.25
    h_far = float(argv[2]) if len(argv) > 2 else 4.0
    threads = int(argv[3]) if len(argv) > 3 else 4
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add("sst")
    try:
        mesh_domain(build_aircraft(), out, h_body, h_far, threads)
    finally:
        gmsh.finalize()


if __name__ == "__main__":
    main()
