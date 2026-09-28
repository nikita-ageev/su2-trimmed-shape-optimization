"""DAKOTA analysis driver for the trimmed shape-optimisation problem of driver.py.

DAKOTA (https://dakota.sandia.gov, open source) runs the optimisation and calls this script once per design:

    python scripts/dakota_driver.py --workdir runs/dk <parameters_file> <results_file>

The parameters file (standard or aprepro format, see "Parameters file format" in the DAKOTA reference manual)
carries the optimiser variables p of driver.Problem (FFD control-point displacements, in the order of
Problem.pvars), the active-set vector ASV (1 = value, 2 = gradient, 4 = Hessian) and the derivative variables
DVV. The results file returns, in the order of the DAKOTA response set,

    f = -K = -CL/CD          objective function (DAKOTA minimises)
    g = vol_frac - V/V0      nonlinear inequality constraint, g <= 0 (DAKOTA's default bounds)
    h = CMy                  nonlinear equality constraint, target 0 (trim)

and, when the ASV asks for them, the gradients with respect to the DVV variables: df/dp = (CL/CD^2) dCD/dp
with dCD/dp from the fixed-CL discrete adjoint, dh/dp from the second adjoint (CMy) and dg/dp = -d(V/V0)/dp
from the analytic volume gradient (driver.Evaluator.gradients, su2run.Volume). Hessians are not available:
use a quasi-Newton or SQP method.

Each call is a new process. The driver reloads the designs already computed in the workdir (folders dsn_NNN),
answers from that cache when DAKOTA asks again for a design it has seen (e.g. the gradient after the value)
and restarts SU2 from the last converged solution. Evaluations therefore run one at a time (DAKOTA's default,
no `asynchronous`); a lock file enforces it.

Set-up, once, from the repository root (embeds the FFD box into the mesh, as driver.py does, and writes a
DAKOTA input file with the right number of variables, bounds and, for the B-spline box, the linear
smoothness constraints):

    python scripts/dakota_driver.py setup --mesh mesh/wing_body.su2 --workdir runs/dk --ffd bezier \\
        --out dakota/trimmed_ld.in
    dakota -i dakota/trimmed_ld.in -o runs/dk/dakota.out

`--stub` replaces SU2 by an analytic test function with the same interface (tests/test_dakota_driver.py, and a
way to try a DAKOTA input file without SU2): `setup --stub --out stub.in` followed by `dakota -i stub.in`.
"""
import argparse
import fcntl
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RESPONSES = ("minus_K", "vol_deficit", "CMy")
HEADERS = {"variables": "vars", "DAKOTA_VARS": "vars", "functions": "asv", "DAKOTA_FNS": "asv",
           "derivative_variables": "dvv", "DAKOTA_DER_VARS": "dvv", "analysis_components": "ac",
           "DAKOTA_AN_COMPS": "ac", "eval_id": "eval_id", "DAKOTA_EVAL_ID": "eval_id",
           "metadata": "md", "DAKOTA_METADATA": "md"}
METHODS = {
    "optpp_q_newton": ["optpp_q_newton", "  max_iterations = {maxiter}", "  max_function_evaluations = {maxfev}",
                       "  convergence_tolerance = 1.0e-6", "  gradient_tolerance = 1.0e-4", "  max_step = 0.3",
                       "  merit_function el_bakry", "  scaling"],
    "rol": ["rol", "  max_iterations = {maxiter}", "  gradient_tolerance = 1.0e-4", "  constraint_tolerance = 1.0e-6",
            "  variable_tolerance = 1.0e-6", "  scaling"],
}


# ---------------------------------------------------------------------------- DAKOTA file formats

class Params:
    """Contents of a DAKOTA parameters file, standard or aprepro format."""

    def __init__(self, path):
        self.x, self.var_tags, self.asv, self.fn_tags, self.dvv, self.ac, self.md = [], [], [], [], [], [], []
        self.eval_id, self.aprepro = "", False
        section, remaining = None, 0
        with open(path) as f:
            lines = f.read().splitlines()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):                                   # aprepro: { tag = value }
                self.aprepro = True
                tag, value = [t.strip() for t in line.strip("{} \t").split("=", 1)]
            else:                                                      # standard: value tag
                value, tag = line.split(None, 1)
            if remaining == 0:
                section = HEADERS.get(tag)
                if section is None:
                    raise ValueError(f"{path}: unexpected line '{line}'")
                if section == "eval_id":
                    self.eval_id = value
                else:
                    remaining = int(value)
                continue
            remaining -= 1
            if section == "vars":
                self.x.append(float(value))
                self.var_tags.append(tag)
            elif section == "asv":
                self.asv.append(int(value))
                self.fn_tags.append(tag.split(":", 1)[1])
            elif section == "dvv":
                self.dvv.append(int(value))
            elif section == "ac":
                self.ac.append(value)
            elif section == "md":
                self.md.append(tag.split(":", 1)[1])
        if remaining:
            raise ValueError(f"{path}: truncated section '{section}'")
        if not self.dvv:
            self.dvv = list(range(1, len(self.x) + 1))

    @property
    def need_grad(self):
        return any(a & 2 for a in self.asv)


def write_results(path, prm, vals, grads):
    """Results file: the requested function values (one per line, with the DAKOTA descriptor), then the
    requested gradients in [ ], restricted to the DVV variables."""
    idx = np.asarray(prm.dvv, dtype=int) - 1
    with open(path, "w") as f:
        for a, v, tag in zip(prm.asv, vals, prm.fn_tags):
            if a & 1:
                f.write(f"{v:.16e} {tag}\n")
        for a, g in zip(prm.asv, grads if grads is not None else [None] * len(prm.asv)):
            if a & 2:
                f.write("[ " + " ".join(f"{v:.16e}" for v in np.asarray(g)[idx]) + " ]\n")


# ---------------------------------------------------------------------------- responses

def responses(P, E, p, need_grad):
    """(values, gradients) of (-K, vol_frac - V/V0, CMy) at p, from a driver.Problem / driver.Evaluator pair
    or their stub counterparts. Gradients are None unless requested."""
    p = np.asarray(p, dtype=float)
    if len(p) != P.np:
        raise SystemExit(f"parameters file has {len(p)} variables, the problem has {P.np}")
    r = E.gradients(p) if need_grad else E.primal(p)
    vals = [-r["CL"] / r["CD"], P.settings["vol_frac"] - P.v_rel(p), r["CMy"]]
    grads = None
    if need_grad:
        grads = [(r["CL"] / r["CD"] ** 2) * r["gCD"], -P.dc_vol(p) / 100.0, r["gCMy"]]
    return vals, grads


class StubProblem:
    """Analytic stand-in for driver.Problem (no SU2, no mesh): n bounded variables, a convex CD, a linear
    CMy and a linear volume, so that a DAKOTA input file and the driver can be exercised in seconds."""

    def __init__(self, n, vol_frac=0.995):
        self.np = n
        self.settings = dict(vol_frac=vol_frac, target_cl=0.10, ffd="stub")
        self.pvars = [("z", i, 0, 0) for i in range(n)]
        self.Lsm, self.Dsm = np.zeros(0), np.zeros((0, n))
        k = np.arange(1, n + 1, dtype=float)
        self.a = 0.3 * np.cos(k)                 # unconstrained minimiser of CD
        self.w = 0.05 * np.sin(k)                # dCMy/dp
        self.v = 0.02 * np.ones(n) / np.sqrt(n)  # d(V/V0)/dp

    def bounds(self):
        return [(-1.0, 1.0)] * self.np

    def v_rel(self, p):
        return 0.99 + self.v @ np.asarray(p, dtype=float)

    def c_vol(self, p):
        return 100.0 * (self.v_rel(p) - self.settings["vol_frac"])

    def dc_vol(self, p):
        return 100.0 * self.v


class StubEvaluator:
    def __init__(self, P):
        self.P = P

    def primal(self, p):
        P, p = self.P, np.asarray(p, dtype=float)
        return dict(p=p, CD=0.005 + 0.002 * np.sum((p - P.a) ** 2), CL=P.settings["target_cl"],
                    CMy=P.w @ p - 0.004, AoA=2.0, Vrel=P.v_rel(p))

    def gradients(self, p):
        r = self.primal(p)
        r["gCD"], r["gCMy"] = 0.004 * (r["p"] - self.P.a), self.P.w.copy()
        return r


def load_problem(workdir, stub_n=None):
    """(Problem, Evaluator) for the workdir, or the stub pair with stub_n variables."""
    if stub_n is not None:
        P = StubProblem(stub_n)
        return P, StubEvaluator(P)
    import su2run as S
    from driver import Problem, Evaluator
    S.check_su2()
    P = Problem.load(workdir)
    return P, Evaluator(P).restore()


# ---------------------------------------------------------------------------- DAKOTA input file

def _fmt_list(values, per_line=10, fmt="{:.6g}"):
    vals = [fmt.format(v) for v in values]
    return "\n".join("      " + " ".join(vals[i:i + per_line]) for i in range(0, len(vals), per_line))


def write_input(path, P, workdir, driver_cmd, method="optpp_q_newton", maxiter=30, stub=False):
    """DAKOTA input for the problem P: variables with bounds and descriptors, linear smoothness constraints
    (B-spline box), the fork interface to this driver and the three scaled responses."""
    n = P.np
    lo, hi = zip(*P.bounds())
    names = ["_".join(str(t) for t in v) for v in P.pvars]
    wd = workdir.replace("\\", "/")
    L = [f"# DAKOTA input written by scripts/dakota_driver.py setup: ffd = {P.settings['ffd']}, {n} variables.",
         "# Run from the repository root:  dakota -i <this file> -o <workdir>/dakota.out",
         "environment", "  tabular_data", f"    tabular_data_file = '{wd}/dakota_tabular.dat'",
         "  top_method_pointer = 'OPT'", "", "method", "  id_method = 'OPT'"]
    L += ["  " + s.format(maxiter=maxiter, maxfev=5 * maxiter) for s in METHODS[method]]
    L += ["  output normal", "", "variables", f"  continuous_design = {n}",
          "    initial_point", _fmt_list([0.0] * n), "    lower_bounds", _fmt_list(lo), "    upper_bounds",
          _fmt_list(hi), "    descriptors", _fmt_list([f"'{s}'" for s in names], fmt="{}")]
    if len(P.Lsm):
        L += [f"  linear_inequality_constraint_matrix   # {len(P.Lsm)} smoothness rows: |D p| <= s",
              _fmt_list(P.Dsm.ravel(), per_line=n, fmt="{:.10g}"), "  linear_inequality_lower_bounds",
              _fmt_list(-P.Lsm), "  linear_inequality_upper_bounds", _fmt_list(P.Lsm)]
    L += ["", "interface", f"  analysis_drivers = '{driver_cmd}'", "    fork",
          f"      parameters_file = '{wd}/params.in'", f"      results_file = '{wd}/results.out'",
          "      file_save", "", "responses",
          "  objective_functions = 1", "    primary_scale_types = 'value'", "    primary_scales = 20.0",
          "  nonlinear_inequality_constraints = 1", "    nonlinear_inequality_scale_types = 'value'",
          "    nonlinear_inequality_scales = 0.01",
          "  nonlinear_equality_constraints = 1", "    nonlinear_equality_scale_types = 'value'",
          "    nonlinear_equality_scales = 1.0e-3",
          "  descriptors " + " ".join(f"'{s}'" for s in RESPONSES),
          "  analytic_gradients", "  no_hessians", ""]
    if stub:
        L[0] = f"# DAKOTA input written by scripts/dakota_driver.py setup --stub: analytic test problem, {n} variables."
    with open(path, "w") as f:
        f.write("\n".join(L))


# ---------------------------------------------------------------------------- command line

def run_driver(a):
    prm = Params(a.params)
    if any(v & 4 for v in prm.asv):
        raise SystemExit("Hessians (ASV bit 4) are not available: use a quasi-Newton or SQP method")
    if len(prm.asv) != len(RESPONSES):
        raise SystemExit(f"expected {len(RESPONSES)} response functions {RESPONSES}, got {len(prm.asv)}")
    if a.stub:
        P, E = load_problem(None, stub_n=len(prm.x))
        vals, grads = responses(P, E, prm.x, prm.need_grad)
    else:
        if not a.workdir:
            raise SystemExit("--workdir is required (see 'setup')")
        with open(os.path.join(a.workdir, "dakota_driver.lock"), "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            P, E = load_problem(a.workdir)
            vals, grads = responses(P, E, prm.x, prm.need_grad)
            with open(os.path.join(a.workdir, "dakota_evals.csv"), "a") as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')},{prm.eval_id},{E.find(np.asarray(prm.x))['n']},"
                        f"{'+'.join(str(v) for v in prm.asv)},{','.join(f'{v:.8g}' for v in vals)}\n")
    write_results(a.results, prm, vals, grads)


def run_setup(a):
    if a.stub:
        P = StubProblem(a.stub_n)
        driver = f"{a.python} {os.path.abspath(__file__)} --stub" if a.driver_cmd is None else a.driver_cmd
    else:
        import su2run as S
        from driver import Problem
        S.check_su2()
        P = Problem(a.workdir, mesh=a.mesh, ffd=a.ffd, target_cl=a.target_cl, xcg=a.xcg, vol_frac=a.vol_frac)
        driver = f"{a.python} scripts/dakota_driver.py --workdir {a.workdir}" if a.driver_cmd is None \
            else a.driver_cmd
    os.makedirs(a.workdir, exist_ok=True)
    write_input(a.out, P, a.workdir, driver, method=a.method, maxiter=a.maxiter, stub=a.stub)
    print(f"{a.out}: {P.np} variables, {len(P.Lsm)} linear constraints, method {a.method}")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] == "setup":
        ap = argparse.ArgumentParser(prog="dakota_driver.py setup",
                                     description="embed the FFD box and write a DAKOTA input file")
        ap.add_argument("--workdir", default="runs/dakota")
        ap.add_argument("--mesh", help="input SU2 mesh (needed for a new workdir)")
        ap.add_argument("--out", default="dakota/trimmed_ld.in", help="DAKOTA input file to write")
        ap.add_argument("--ffd", default="bezier", help="FFD preset of driver.py: bezier (66 variables) or bspline")
        ap.add_argument("--method", choices=sorted(METHODS), default="optpp_q_newton")
        ap.add_argument("--maxiter", type=int, default=30)
        ap.add_argument("--target-cl", type=float, default=0.10)
        ap.add_argument("--xcg", type=float, default=None)
        ap.add_argument("--vol-frac", type=float, default=0.995)
        ap.add_argument("--python", default="python", help="interpreter in the analysis_drivers string")
        ap.add_argument("--driver-cmd", default=None, help="full analysis_drivers string (overrides --python)")
        ap.add_argument("--stub", action="store_true", help="analytic test problem instead of SU2")
        ap.add_argument("--stub-n", type=int, default=6, help="number of variables of the stub problem")
        a = ap.parse_args(argv[1:])
        if a.xcg is None:
            import geometry as G
            a.xcg = G.X_CG
        run_setup(a)
    else:
        ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
        ap.add_argument("params", help="parameters file written by DAKOTA")
        ap.add_argument("results", help="results file read by DAKOTA")
        ap.add_argument("--workdir", default=None, help="workdir prepared by 'setup' (or by driver.py)")
        ap.add_argument("--stub", action="store_true", help="analytic test problem instead of SU2")
        run_driver(ap.parse_args(argv))


if __name__ == "__main__":
    main()
