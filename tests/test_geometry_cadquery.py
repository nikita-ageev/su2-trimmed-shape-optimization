"""Tests of the CadQuery geometry route. The analytic parts run everywhere; the CAD parts are skipped without
cadquery, the mesh part without gmsh (both are optional dependencies, see README).

Run from the repository root:  python -m unittest discover -s tests -v
"""
import math
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import geometry as G  # noqa: E402
import geometry_cadquery as GC  # noqa: E402

try:
    import cadquery  # noqa: F401
    HAVE_CQ = True
except ImportError:
    HAVE_CQ = False
try:
    import gmsh  # noqa: F401
    HAVE_GMSH = True
except ImportError:
    HAVE_GMSH = False


class TestRadiusLaw(unittest.TestCase):
    def test_sears_haack_volume_matches_geometry_py(self):
        # V = 3 pi^2 / 16 R^2 L for n = 0.75
        self.assertAlmostEqual(GC.body_volume(G.L_FUS, G.R_MAX), G.V_FUS_FULL, delta=1e-8 * G.V_FUS_FULL)
        self.assertAlmostEqual(GC.r_max_for_volume(G.L_FUS, G.V_FUS_FULL), G.R_MAX, places=9)

    def test_base_cut_and_exponent(self):
        v_full = GC.body_volume(10.0, 1.0, n=0.6)
        v_cut = GC.body_volume(10.0, 1.0, n=0.6, base=0.9)
        self.assertLess(v_cut, v_full)
        self.assertGreater(v_cut, 0.9 * v_full)
        r = GC.r_max_for_volume(10.0, v_cut, n=0.6, base=0.9)
        self.assertAlmostEqual(r, 1.0, places=9)
        pts = GC.meridian(10.0, 1.0, n=0.6, base=0.9)
        self.assertEqual(pts[0], (0.0, 0.0))
        self.assertAlmostEqual(pts[-1][0], 9.0)
        self.assertGreater(pts[-1][1], 0.0)


@unittest.skipUnless(HAVE_CQ, "cadquery not installed")
class TestCadQuery(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="cq_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_fuselage_volume(self):
        v = GC.fuselage().val().Volume()
        self.assertAlmostEqual(v, G.V_FUS_FULL, delta=2e-4 * G.V_FUS_FULL)     # spline vs analytic
        v_cut = GC.fuselage(30.0, 100.0, n=0.6, base=0.9).val().Volume()
        self.assertAlmostEqual(v_cut, 100.0, delta=2e-4 * 100.0)

    def test_aircraft_step_roundtrip(self):
        import cadquery as cq
        shape = GC.aircraft()
        self.assertEqual(len(shape.solids().vals()), 1)
        v = shape.val().Volume()
        self.assertGreater(v, G.V_FUS_FULL)                                    # wing adds volume
        step = GC.write_step(shape, os.path.join(self.tmp, "aircraft.step"))
        back = cq.importers.importStep(step)
        self.assertEqual(len(back.solids().vals()), 1)
        self.assertAlmostEqual(back.val().Volume(), v, delta=1e-6 * v)
        bb = back.val().BoundingBox()
        self.assertAlmostEqual(bb.xmin, 0.0, places=5)
        self.assertAlmostEqual(bb.xmax, G.L_FUS, places=5)
        self.assertAlmostEqual(bb.ymax, G.SEMI_SPAN, places=5)

    @unittest.skipUnless(HAVE_GMSH, "gmsh not installed")
    def test_mesh_from_step(self):
        import su2run as S
        step = GC.write_step(GC.aircraft(), os.path.join(self.tmp, "aircraft.step"))
        out = os.path.join(self.tmp, "wb.su2")
        ntet, nodes = GC.mesh_from_step(step, out, h_body=1.5, h_far=8.0, threads=2)
        self.assertGreater(ntet, 1000)
        with open(out) as f:
            txt = f.read()
        markers = [l.split("=")[1].strip() for l in txt.splitlines() if l.startswith("MARKER_TAG")]
        self.assertEqual(sorted(markers), ["aircraft", "farfield", "symmetry"])
        c, t = S.read_su2_marker(out, "aircraft", with_tets=True)
        v_half = abs(S.half_volume(c, t))
        # half fuselage + the wing panel at y > 0, on a coarse surface mesh
        self.assertGreater(v_half, 0.85 * G.V_FUS_HALF)
        self.assertLess(v_half, 1.8 * G.V_FUS_HALF)
        self.assertTrue(np.isfinite(v_half))
        surf = c[np.unique(t)]                       # read_su2_marker returns all nodes (global numbering)
        self.assertAlmostEqual(surf[:, 0].min(), 0.0, delta=1e-3)
        self.assertAlmostEqual(surf[:, 0].max(), G.L_FUS, delta=1e-3)
        self.assertGreaterEqual(surf[:, 1].min(), -1e-6)
        self.assertAlmostEqual(surf[:, 1].max(), G.SEMI_SPAN, delta=1e-3)


if __name__ == "__main__":
    unittest.main()
