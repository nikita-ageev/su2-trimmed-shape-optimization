"""Tests of the v0.3 robust preset (no SU2 needed): numerics in the SU2 config, clamped nose/tail planes,
the Sobolev metric and the smoothness metrics on synthetic bodies.

Run from the repository root:  python -m unittest discover -s tests -v     (or: pytest tests)
"""
import os
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import presets as PR  # noqa: E402
import smoothness as SM  # noqa: E402
import su2run as S  # noqa: E402
import geometry as G  # noqa: E402


def pvars_for(L, M=4, N=4, clamp=()):
    """Variables of driver.Problem for a (L+1) x (M+1) x (N+1) box with fixj = M and linked z of j = 0, 1."""
    free = [i for i in range(L + 1) if i not in clamp]
    return [("z", i, j, k) for i in free for j in range(1, M) for k in range(N + 1)] + \
           [("y", i, j, k) for i in free for j in range(1, M) for k in range(N + 1)]


class TestNumerics(unittest.TestCase):
    def fill(self, numerics):
        """Fill config/wing_body.cfg the way driver.Problem.cfg does and return the text."""
        spec = S.FFDSpec("bspline_low", G.FFD_PRESETS["bspline_low"])
        d = dict(MATH_PROBLEM="DIRECT", RESTART_SOL="NO", AOA="2.0", DCD_DCL="0.0", DCMY_DCL="0.0", FIXED_CL="YES",
                 EVAL_DOF_DCX="NO", DISCARD_INFILES="NO", XCG="42.35", TARGET_CL="0.1", OBJ="DRAG", ITER="10",
                 CONV_FIELD="RMS_DENSITY", MINVAL="-8", MESH_IN="m.su2", MESH_OUT="o.su2", SOLUTION="s.dat",
                 RESTART="r.dat", OUTPUT_FILES="(RESTART)", CONV_FILENAME="history", SURFACE_FILENAME="surface_flow",
                 SURFACE_ADJ_FILENAME="surface_adjoint", GRAD_FILENAME="of_grad.csv", SCREEN_OUTPUT="(INNER_ITER)",
                 HISTORY_OUTPUT="(ITER)", NUMERICS=PR.NUMERICS[numerics])
        d.update(spec.cfg_fields())
        d.update(spec.dv_strings(np.zeros(spec.ndv)))
        with tempfile.TemporaryDirectory() as t:
            out = os.path.join(t, "c.cfg")
            S.fill_template(os.path.join(ROOT, "config", "wing_body.cfg"), d, out)
            with open(out) as fh:
                return fh.read()

    def test_first_order(self):
        txt = self.fill("first_order")
        self.assertIn("CONV_NUM_METHOD_FLOW= ROE", txt)
        self.assertIn("MUSCL_FLOW= NO", txt)
        self.assertNotIn("JST", txt.split("% spatial scheme")[1].split("\n")[1])

    def test_second_order_is_v02(self):
        txt = self.fill("second_order")
        self.assertIn("CONV_NUM_METHOD_FLOW= JST", txt)
        self.assertIn("JST_SENSOR_COEFF= ( 0.5, 0.02 )", txt)

    def test_presets(self):
        r = PR.problem_kwargs("robust")
        self.assertEqual(r["numerics"], "first_order")
        self.assertEqual(r["ffd"], "bspline_low")
        self.assertEqual(tuple(r["clamp_i"]), (0, 1, G.FFD_PRESETS["bspline_low"]["deg"][0]))
        self.assertEqual(PR.problem_kwargs("verify")["numerics"], "second_order")
        self.assertIsNone(PR.problem_kwargs("v02")["sobolev"])


class TestClampAndMetric(unittest.TestCase):
    def test_clamped_planes_have_no_variables(self):
        pv = pvars_for(8, clamp=(0, 1, 8))
        self.assertFalse(any(v[1] in (0, 1, 8) for v in pv))
        self.assertEqual(len(pv), 6 * 3 * 5 * 2)

    def test_metric_identity_without_preset(self):
        pv = pvars_for(8)
        self.assertTrue(np.array_equal(PR.sobolev_metric(pv, None), np.eye(len(pv))))

    def test_metric_penalises_zigzag(self):
        pv = pvars_for(8, clamp=(0, 1, 8))
        M = PR.sobolev_metric(pv, (2.0, 0.5))
        self.assertTrue(np.allclose(M, M.T))
        self.assertGreater(np.linalg.eigvalsh(M).min(), 0.999)          # SPD, >= I
        free = sorted({v[1] for v in pv})
        pos = {i: m for m, i in enumerate(free)}
        # every z control point of plane i moves by the same amount: zig-zag vs a smooth bend along x
        zig = np.array([(-1.0) ** pos[v[1]] if v[0] == "z" else 0.0 for v in pv])
        bend = np.array([np.sin(0.2 + (np.pi - 0.4) * pos[v[1]] / (len(free) - 1)) if v[0] == "z" else 0.0 for v in pv])
        cost = lambda v: v @ M @ v / (v @ v)  # noqa: E731
        self.assertGreater(cost(zig), 10 * cost(bend))

    def test_smoothed_step_is_smooth(self):
        """The first trust-region step -H^-1 g with H = M turns a zig-zag gradient into a smooth step."""
        pv = pvars_for(8, clamp=(0, 1, 8))
        M = PR.sobolev_metric(pv, (2.0, 0.5))
        line = [n for n, v in enumerate(pv) if v[0] == "z" and v[2] == 2 and v[3] == 2]
        g = np.zeros(len(pv)); g[line] = 1.0 + 0.5 * (-1.0) ** np.arange(len(line))
        d = -np.linalg.solve(M, g)[line]
        zig_in, zig_out = np.abs(np.diff(g[line], 2)).max(), np.abs(np.diff(d, 2)).max()
        self.assertLess(zig_out / np.abs(d).max(), 0.2 * zig_in / np.abs(g[line]).max())


def write_body(path_mesh, path_csv, r_of_x, L=10.0, nx=161, nphi=25, pinf=1e4, mach=2.0):
    """Half body of revolution (y >= 0) as a triangulated marker `aircraft` plus a surface_flow.csv whose
    pressure follows the local slope (Cp = 2 dr/dx, linear theory), so warts show in Cp too."""
    xs = np.linspace(0.0, L, nx)
    ph = np.linspace(0.0, np.pi, nphi)
    r = r_of_x(xs)
    X = np.array([[x, rr * np.sin(p), rr * np.cos(p)] for x, rr in zip(xs, r) for p in ph])
    idx = lambda i, j: i * nphi + j  # noqa: E731
    tris = []
    for i in range(nx - 1):
        for j in range(nphi - 1):
            tris += [(idx(i, j), idx(i + 1, j), idx(i + 1, j + 1)), (idx(i, j), idx(i + 1, j + 1), idx(i, j + 1))]
    with open(path_mesh, "w") as f:
        f.write("NDIME= 3\nNMARK= 1\nMARKER_TAG= aircraft\n")
        f.write(f"MARKER_ELEMS= {len(tris)}\n")
        for t in tris:
            f.write(f"5 {t[0]} {t[1]} {t[2]}\n")
    cp = np.repeat(2.0 * np.gradient(r, xs), nphi)
    gamma, rho, a = 1.4, 0.2, 300.0
    p = pinf * (1 + cp * 0.5 * gamma * mach ** 2)
    u = mach * np.sqrt(gamma * pinf / rho)
    E = p / (gamma - 1) + 0.5 * rho * u ** 2
    with open(path_csv, "w") as f:
        f.write('"PointID","x","y","z","Density","Momentum_x","Momentum_y","Momentum_z","Energy"\n')
        for n, (q, e) in enumerate(zip(X, E)):
            f.write(f"{n}, {q[0]:.9e}, {q[1]:.9e}, {q[2]:.9e}, {rho:.9e}, {rho * u:.9e}, 0.0, 0.0, {e:.9e}\n")


class TestSmoothness(unittest.TestCase):
    L = 10.0

    def metrics(self, r_base, r_final):
        with tempfile.TemporaryDirectory() as t:
            mesh = os.path.join(t, "m.su2")
            for name, rf in (("b", r_base), ("f", r_final)):
                os.makedirs(os.path.join(t, name))
                write_body(mesh, os.path.join(t, name, "surface_flow.csv"), rf, self.L)
            return SM.evaluate(mesh, os.path.join(t, "b"), os.path.join(t, "f"), zone=(0.2, 4.5), h=0.1, L=self.L,
                               pinf=1e4, mach=2.0)

    def ogive(self, x):
        return 0.6 * (1 - np.clip(1 - x / 5.0, 0, 1) ** 2)

    def test_smooth_nose_has_no_warts(self):
        m = self.metrics(self.ogive, self.ogive)
        t = SM.totals(m, "base")
        self.assertEqual(t["warts"], 0)
        self.assertEqual(SM.totals(m, "final")["d_ext"], 0)

    def test_wart_is_found(self):
        wart = lambda x: self.ogive(x) + 0.05 * np.exp(-((x - 2.0) / 0.15) ** 2)  # noqa: E731
        m = self.metrics(self.ogive, wart)
        t = SM.totals(m, "final")
        self.assertGreaterEqual(t["warts"], 5)                 # one bump on each of the five rays
        self.assertGreater(t["cp_ext"], SM.totals(m, "base")["cp_ext"])
        self.assertAlmostEqual(t["d_max_mm"], 50.0, delta=5.0)

    def test_waves_are_counted(self):
        wavy = lambda x: self.ogive(x) + 0.01 * np.sin(2 * np.pi * x / 0.8)  # noqa: E731
        m = self.metrics(self.ogive, wavy)
        self.assertGreater(SM.totals(m, "final")["d_ext"], 5 * 4)


if __name__ == "__main__":
    unittest.main()
