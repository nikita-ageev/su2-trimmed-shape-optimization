"""Tests of the DAKOTA analysis driver on its file formats and on the assembly of the responses, without SU2
(the --stub problem). If a `dakota` executable is on PATH or given in $DAKOTA_EXE, the last test also runs
DAKOTA itself on the stub problem and compares the optimum with SciPy SLSQP.

Run from the repository root:  python -m unittest discover -s tests -v     (or: pytest tests)
"""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import dakota_driver as D  # noqa: E402

DAKOTA = os.environ.get("DAKOTA_EXE") or shutil.which("dakota")


def params_standard(x, asv, dvv, eval_id=7, extra_metadata=True):
    L = [f"{len(x)} variables"] + [f"{v:.15e} x{i + 1}" for i, v in enumerate(x)]
    L += [f"{len(asv)} functions"] + [f"{a} ASV_{i + 1}:{tag}" for i, (a, tag) in enumerate(zip(asv, D.RESPONSES))]
    L += [f"{len(dvv)} derivative_variables"] + [f"{d} DVV_{i + 1}:x{d}" for i, d in enumerate(dvv)]
    L += ["0 analysis_components", f"{eval_id} eval_id"]
    if extra_metadata:
        L += ["0 metadata"]
    return "\n".join(f"{s:>60}" for s in L) + "\n"


def params_aprepro(x, asv, dvv, eval_id=7):
    L = [f"DAKOTA_VARS = {len(x)}"] + [f"x{i + 1} = {v:.15e}" for i, v in enumerate(x)]
    L += [f"DAKOTA_FNS = {len(asv)}"] + [f"ASV_{i + 1}:{tag} = {a}" for i, (a, tag) in enumerate(zip(asv, D.RESPONSES))]
    L += [f"DAKOTA_DER_VARS = {len(dvv)}"] + [f"DVV_{i + 1}:x{d} = {d}" for i, d in enumerate(dvv)]
    L += ["DAKOTA_AN_COMPS = 0", f"DAKOTA_EVAL_ID = {eval_id}", "DAKOTA_METADATA = 0"]
    return "".join("{ " + s + " }\n" for s in L)


def read_results(path):
    vals, grads = [], []
    with open(path) as f:
        lines = f.read().splitlines()
    for line in lines:
        line = line.strip()
        if line.startswith("["):
            grads.append([float(v) for v in line.strip("[] ").split()])
        elif line:
            v, tag = line.split()
            vals.append((float(v), tag))
    return vals, grads


class TestDakotaDriver(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="dk_")
        self.x = np.array([0.4, -0.2, 0.1, 0.0, 0.3, -0.5])
        self.P, self.E = D.load_problem(None, stub_n=len(self.x))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_driver(self, text, argv_extra=()):
        pf, rf = os.path.join(self.tmp, "params.in"), os.path.join(self.tmp, "results.out")
        with open(pf, "w") as f:
            f.write(text)
        D.main([pf, rf, "--stub", *argv_extra])
        return read_results(rf)

    def test_parse_standard_and_aprepro(self):
        for text, aprepro in ((params_standard(self.x, [3, 1, 2], [1, 3, 5]), False),
                              (params_aprepro(self.x, [3, 1, 2], [1, 3, 5]), True)):
            pf = os.path.join(self.tmp, "p.in")
            with open(pf, "w") as f:
                f.write(text)
            prm = D.Params(pf)
            self.assertEqual(prm.aprepro, aprepro)
            np.testing.assert_allclose(prm.x, self.x, rtol=0, atol=1e-15)
            self.assertEqual(prm.var_tags, [f"x{i + 1}" for i in range(6)])
            self.assertEqual(prm.asv, [3, 1, 2])
            self.assertEqual(prm.fn_tags, list(D.RESPONSES))
            self.assertEqual(prm.dvv, [1, 3, 5])
            self.assertEqual(prm.eval_id, "7")
            self.assertTrue(prm.need_grad)

    def test_values_only(self):
        vals, grads = self.run_driver(params_standard(self.x, [1, 1, 1], [1, 2, 3, 4, 5, 6]))
        self.assertEqual(grads, [])
        self.assertEqual([t for _, t in vals], list(D.RESPONSES))
        ref, _ = D.responses(self.P, self.E, self.x, False)
        np.testing.assert_allclose([v for v, _ in vals], ref, rtol=1e-14)
        r = self.E.primal(self.x)
        self.assertAlmostEqual(vals[0][0], -r["CL"] / r["CD"], places=14)
        self.assertAlmostEqual(vals[1][0], 0.995 - self.P.v_rel(self.x), places=14)
        self.assertAlmostEqual(vals[2][0], r["CMy"], places=14)

    def test_gradients_on_dvv_subset(self):
        dvv = [1, 3, 5]
        vals, grads = self.run_driver(params_aprepro(self.x, [3, 3, 3], dvv))
        self.assertEqual(len(vals), 3)
        self.assertEqual([len(g) for g in grads], [3, 3, 3])
        h = 1e-6
        for k, j in enumerate(dvv):
            e = np.zeros(6); e[j - 1] = h
            fp, _ = D.responses(self.P, self.E, self.x + e, False)
            fm, _ = D.responses(self.P, self.E, self.x - e, False)
            for i in range(3):
                self.assertAlmostEqual(grads[i][k], (fp[i] - fm[i]) / (2 * h), delta=1e-7 * max(1, abs(grads[i][k])))

    def test_gradient_only_and_mixed_asv(self):
        vals, grads = self.run_driver(params_standard(self.x, [2, 2, 2], [1, 2, 3, 4, 5, 6]))
        self.assertEqual(vals, [])
        self.assertEqual([len(g) for g in grads], [6, 6, 6])
        vals, grads = self.run_driver(params_standard(self.x, [1, 3, 2], [2, 4]))
        self.assertEqual([t for _, t in vals], ["minus_K", "vol_deficit"])
        self.assertEqual([len(g) for g in grads], [2, 2])

    def test_cli_subprocess(self):
        pf, rf = os.path.join(self.tmp, "params.in"), os.path.join(self.tmp, "results.out")
        with open(pf, "w") as f:
            f.write(params_standard(self.x, [3, 3, 3], [1, 2, 3, 4, 5, 6]))
        subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "dakota_driver.py"), "--stub", pf, rf],
                       check=True)
        vals, grads = read_results(rf)
        self.assertEqual(len(vals), 3)
        self.assertEqual(len(grads), 3)

    def test_rejects_hessians_and_wrong_response_count(self):
        with self.assertRaises(SystemExit):
            self.run_driver(params_standard(self.x, [7, 1, 1], [1]))
        text = params_standard(self.x, [1, 1], [1]).replace("2 functions", "2 functions")
        with self.assertRaises(SystemExit):
            self.run_driver(text)

    def test_setup_writes_input(self):
        out = os.path.join(self.tmp, "stub.in")
        D.main(["setup", "--stub", "--stub-n", "6", "--workdir", self.tmp, "--out", out])
        with open(out) as f:
            txt = f.read()
        for s in ("continuous_design = 6", "analytic_gradients", "no_hessians", "optpp_q_newton",
                  "nonlinear_equality_constraints = 1", "nonlinear_inequality_constraints = 1",
                  "'minus_K' 'vol_deficit' 'CMy'", "--stub"):
            self.assertIn(s, txt)

    @unittest.skipUnless(DAKOTA, "dakota executable not found (PATH or $DAKOTA_EXE)")
    def test_dakota_run_matches_slsqp(self):
        from scipy.optimize import minimize
        n = 6
        out = os.path.join(self.tmp, "stub.in")
        # DAKOTA's fork interface splits the driver string on blanks: run a copy of the driver from the
        # temporary directory, whose path has no blanks (the repository path may have)
        drv = shutil.copy(os.path.join(ROOT, "scripts", "dakota_driver.py"), self.tmp)
        D.main(["setup", "--stub", "--stub-n", str(n), "--workdir", self.tmp, "--out", out,
                "--driver-cmd", f"{sys.executable} {drv} --stub", "--maxiter", "100"])
        r = subprocess.run([DAKOTA, "-i", out], cwd=self.tmp, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
        lines = r.stdout.splitlines()
        k = max(i for i, l in enumerate(lines) if l.startswith("<<<<< Best parameters"))
        p_dk = np.array([float(lines[k + 1 + i].split()[0]) for i in range(n)])
        P, E = D.load_problem(None, stub_n=n)
        f = lambda p: D.responses(P, E, p, False)[0]      # noqa: E731
        vals = f(p_dk)
        self.assertLess(abs(vals[2]), 1e-5)                # CMy = 0
        self.assertLess(vals[1], 1e-5)                     # V/V0 >= 0.995
        res = minimize(lambda p: f(p)[0], np.zeros(n), jac=lambda p: D.responses(P, E, p, True)[1][0],
                       method="SLSQP", bounds=P.bounds(),
                       constraints=[dict(type="eq", fun=lambda p: 1e3 * f(p)[2]),
                                    dict(type="ineq", fun=lambda p: -1e2 * f(p)[1])],
                       options=dict(ftol=1e-12, maxiter=200))
        self.assertTrue(res.success, res.message)
        self.assertAlmostEqual(vals[0], f(res.x)[0], delta=1e-4 * abs(vals[0]))
        self.assertLess(np.abs(p_dk - res.x).max(), 2e-3)


if __name__ == "__main__":
    unittest.main()
