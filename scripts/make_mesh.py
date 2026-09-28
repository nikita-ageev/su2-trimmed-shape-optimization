"""Gmsh mesh for the half model "Sears-Haack fuselage + thin wing".

Usage: python scripts/make_mesh.py <out.su2> [h_body] [h_far] [threads]
Unstructured tetrahedral mesh; markers: aircraft, symmetry, farfield.
h_body = 0.25 m gives ~350k tetrahedra (the published run), 0.6 m gives a quick ~50k-cell smoke-test mesh.
"""
import os
import sys
import numpy as np
import gmsh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import geometry as G

out = sys.argv[1] if len(sys.argv) > 1 else "mesh/wing_body.su2"
H_BODY = float(sys.argv[2]) if len(sys.argv) > 2 else 0.25
H_FAR = float(sys.argv[3]) if len(sys.argv) > 3 else 4.0
H_EDGE = 0.45 * H_BODY
THREADS = int(sys.argv[4]) if len(sys.argv) > 4 else 4
if os.path.dirname(out):
    os.makedirs(os.path.dirname(out), exist_ok=True)

gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 1)
gmsh.model.add("sst")
occ = gmsh.model.occ

# ---------- fuselage: body of revolution with the Sears-Haack radius
xs = G.L_FUS * (0.5 - 0.5 * np.cos(np.linspace(0, np.pi, 61)))   # clustered at the ends
pts = [occ.addPoint(x, 0, float(G.sears_haack_r(x))) for x in xs]
prof = occ.addSpline(pts)
axis = occ.addLine(pts[-1], pts[0])
loop = occ.addCurveLoop([prof, axis])
face = occ.addPlaneSurface([loop])
fus = occ.revolve([(2, face)], 0, 0, 0, 1, 0, 0, 2 * np.pi)
fus_vol = [d for d in fus if d[0] == 3]

# ---------- wing: two sections (root just behind the symmetry plane, tip), biconvex profile
def section(x_le, y, c):
    n = 25
    xi = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, n))
    zt = 2 * G.T_C * c * xi * (1 - xi)
    p_le = occ.addPoint(x_le, y, G.Z_WING)
    p_te = occ.addPoint(x_le + c, y, G.Z_WING)
    up = [p_le] + [occ.addPoint(x_le + c * s, y, G.Z_WING + z) for s, z in zip(xi[1:-1], zt[1:-1])] + [p_te]
    lo = [p_te] + [occ.addPoint(x_le + c * s, y, G.Z_WING - z) for s, z in zip(xi[1:-1][::-1], zt[1:-1][::-1])] + [p_le]
    c1 = occ.addSpline(up)
    c2 = occ.addSpline(lo)
    return occ.addWire([c1, c2])

w_root = section(G.X_LE_ROOT - 0.3 * np.tan(np.radians(G.SWEEP_LE_DEG)), -0.3, G.C_ROOT)  # slightly beyond the symmetry plane
w_tip = section(G.X_LE_TIP, G.SEMI_SPAN, G.C_TIP)
wing = occ.addThruSections([w_root, w_tip], makeSolid=True, makeRuled=True)
wing_vol = [d for d in wing if d[0] == 3]

aircraft, _ = occ.fuse(fus_vol, wing_vol)

# ---------- flow domain: box y >= 0; at supersonic speed disturbances do not travel upstream
BX0, BX1, BY1, BZ = -6.0, 72.0, 32.0, 32.0
box = occ.addBox(BX0, 0.0, -BZ, BX1 - BX0, BY1, 2 * BZ)
fluid, _ = occ.cut([(3, box)], aircraft)
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
gmsh.model.addPhysicalGroup(2, body, name="aircraft")
gmsh.model.addPhysicalGroup(2, sym, name="symmetry")
gmsh.model.addPhysicalGroup(2, far, name="farfield")
gmsh.model.addPhysicalGroup(3, [vols[0][1]], name="fluid")

# ---------- cell size: fine at the body, finer at the wing edges, growing towards the far field
body_curves = set()
for s in body:
    for _, c in gmsh.model.getBoundary([(2, s)], oriented=False):
        body_curves.add(abs(c))
f1 = gmsh.model.mesh.field.add("Distance")
gmsh.model.mesh.field.setNumbers(f1, "SurfacesList", body)
gmsh.model.mesh.field.setNumber(f1, "Sampling", 200)
t1 = gmsh.model.mesh.field.add("Threshold")
gmsh.model.mesh.field.setNumber(t1, "InField", f1)
gmsh.model.mesh.field.setNumber(t1, "SizeMin", H_BODY)
gmsh.model.mesh.field.setNumber(t1, "SizeMax", H_FAR)
gmsh.model.mesh.field.setNumber(t1, "DistMin", 0.3)
gmsh.model.mesh.field.setNumber(t1, "DistMax", 28.0)
f2 = gmsh.model.mesh.field.add("Distance")
gmsh.model.mesh.field.setNumbers(f2, "CurvesList", sorted(body_curves))
gmsh.model.mesh.field.setNumber(f2, "Sampling", 400)
t2 = gmsh.model.mesh.field.add("Threshold")
gmsh.model.mesh.field.setNumber(t2, "InField", f2)
gmsh.model.mesh.field.setNumber(t2, "SizeMin", H_EDGE)
gmsh.model.mesh.field.setNumber(t2, "SizeMax", H_FAR)
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
gmsh.option.setNumber("General.NumThreads", THREADS)
gmsh.option.setNumber("Mesh.Optimize", 1)
gmsh.model.mesh.generate(3)
ntet = sum(len(t) for t in gmsh.model.mesh.getElementsByType(4)[:1])
nodes = len(gmsh.model.mesh.getNodes()[0])
print(f"TETS {ntet}  NODES {nodes}")
gmsh.write(out)
gmsh.finalize()
