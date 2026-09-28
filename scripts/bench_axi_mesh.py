"""Axisymmetric 2D mesh around a Sears-Haack body, L = 1, V = 0.01 L^3 (R_max = 0.0735 L).
Markers: body, axis (symmetry axis), farfield.

Usage: python scripts/bench_axi_mesh.py <out.su2> [h_body]   (h_body = 0.003 -> ~25k triangles)
"""
import os
import sys

import numpy as np

L = 1.0
V = 0.01 * L**3
R_MAX = np.sqrt(V * 16 / (3 * np.pi**2 * L))      # V = 3 pi^2 / 16 R^2 L


def make_mesh(out, h=0.003):
    import gmsh
    if os.path.dirname(out):
        os.makedirs(os.path.dirname(out), exist_ok=True)
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    occ = gmsh.model.occ
    xs = L * (0.5 - 0.5 * np.cos(np.linspace(0, np.pi, 81)))
    pb = [occ.addPoint(x, R_MAX * (4 * x / L * (1 - x / L)) ** 0.75, 0) for x in xs]
    body = occ.addSpline(pb)
    X0, X1, Y1 = -0.3, 1.6, 1.0
    p1 = occ.addPoint(X0, 0, 0); p2 = occ.addPoint(X1, 0, 0); p3 = occ.addPoint(X1, Y1, 0); p4 = occ.addPoint(X0, Y1, 0)
    a1 = occ.addLine(p1, pb[0]); a2 = occ.addLine(pb[-1], p2)
    f1 = occ.addLine(p2, p3); f2 = occ.addLine(p3, p4); f3 = occ.addLine(p4, p1)
    loop = occ.addCurveLoop([a1, body, a2, f1, f2, f3])
    s = occ.addPlaneSurface([loop])
    occ.synchronize()
    gmsh.model.addPhysicalGroup(1, [body], name="body")
    gmsh.model.addPhysicalGroup(1, [a1, a2], name="axis")
    gmsh.model.addPhysicalGroup(1, [f1, f2, f3], name="farfield")
    gmsh.model.addPhysicalGroup(2, [s], name="fluid")
    fd = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(fd, "CurvesList", [body])
    gmsh.model.mesh.field.setNumber(fd, "Sampling", 500)
    th = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(th, "InField", fd)
    gmsh.model.mesh.field.setNumber(th, "SizeMin", h)
    gmsh.model.mesh.field.setNumber(th, "SizeMax", 0.04)
    gmsh.model.mesh.field.setNumber(th, "DistMin", 0.01)
    gmsh.model.mesh.field.setNumber(th, "DistMax", 0.8)
    gmsh.model.mesh.field.setAsBackgroundMesh(th)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.model.mesh.generate(2)
    ntri = len(gmsh.model.mesh.getElementsByType(2)[0])
    gmsh.write(out)
    gmsh.finalize()
    print(f"mesh {out}: {ntri} triangles, R_max = {R_MAX:.5f}, L/D = {L / (2 * R_MAX):.2f}")


if __name__ == "__main__":
    make_mesh(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 0.003)
