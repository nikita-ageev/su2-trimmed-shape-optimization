"""Trust-region SQP on top of the driver's Problem/Evaluator (used for the 450-variable B-spline run instead of
SciPy SLSQP, whose line search stalled and whose first step after a restart was far too large).

Iteration k solves the quadratic subproblem

    min_d  g.d + 1/2 d'Hd   s.t.  c_eq + a_eq.d = 0          (CMy = 0, linearised)
                                  c_V + a_V.d >= 0           (volume, linearised)
                                  smoothness rows            (linear, exact)
                                  variable bounds, |d|_inf <= Delta

(solved cheaply by SLSQP on the model), followed by a second-order correction that puts the volume back on its
bound (the volume is exact and cheap). The step is accepted on the merit function
    phi = 1e4 CD + nu |1e3 CMy| + nu max(0, -c_V),
and Delta is updated from the ratio of actual to predicted reduction of phi. H is a BFGS approximation of the
Hessian of the Lagrangian with Powell damping; the multipliers come from sign-constrained least squares on the
KKT conditions, whose relative residual is reported as "kkt".

Stops when K changes by less than --stop-dk on three accepted steps with all constraints satisfied, when the KKT
residual is below --kkt-tol, when Delta < 2 mm, or on the iteration / time budget.

Usage (after driver.py, in the same workdir; starts from the last accepted SLSQP iterate by default):
    SU2_NP=8 python scripts/trsqp.py --workdir runs/wing_body --maxiter 60 [--start-eval N]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.optimize import minimize, lsq_linear

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import su2run as S  # noqa: E402
from driver import Problem, Evaluator, F_SCALE, C_SCALE  # noqa: E402

TOL_ACTIVE = 2e-3


class TRSQP:
    def __init__(self, P, start_eval, nu=20.0, t0=None):
        self.P, self.nu = P, nu
        self.t0 = t0 or time.time()
        self.E = Evaluator(P, t0=self.t0)
        self.Jsm = P.dc_smooth(np.zeros(P.np))       # constant (linear constraints)
        self.p = self.load_eval(start_eval)
        runs = P.runs
        self.E.n = max(int(n.split("_")[1]) for n in os.listdir(runs)
                       if n.startswith("dsn_") and n[4:].isdigit()) + 1

    def load_eval(self, ev):
        """Put an evaluated design into the evaluator cache without recomputing it."""
        return self.E.load_eval(ev)["p"]

    def funcs(self, p, grad=True):
        P, E = self.P, self.E
        r = E.gradients(p) if grad else E.primal(p)
        F = dict(f=F_SCALE * r["CD"], ceq=C_SCALE * r["CMy"], cv=P.c_vol(p), r=r)
        if grad:
            F.update(g=F_SCALE * r["gCD"], aeq=C_SCALE * r["gCMy"], av=P.dc_vol(p))
        return F

    def merit(self, F):
        return F["f"] + self.nu * abs(F["ceq"]) + self.nu * max(0.0, -F["cv"])

    def multipliers(self, F, p):
        """Lagrange multipliers by sign-constrained least squares; returns (lambda_CMy, mu_V, KKT residual)."""
        P = self.P
        cols, lb = [F["aeq"]], [-np.inf]
        vol_active = F["cv"] < TOL_ACTIVE
        if vol_active:
            cols.append(F["av"]); lb.append(0.0)
        if len(P.Lsm):
            for i in np.where(P.c_smooth(p) < TOL_ACTIVE)[0]:
                cols.append(self.Jsm[i]); lb.append(0.0)
        for i, (lo, hi) in enumerate(P.bounds()):
            if p[i] <= lo + 1e-6 or p[i] >= hi - 1e-6:
                e = np.zeros(P.np); e[i] = 1.0 if p[i] <= lo + 1e-6 else -1.0
                cols.append(e); lb.append(0.0)
        A, g = np.array(cols).T, F["g"]
        try:
            xs = lsq_linear(A, g, bounds=(np.array(lb), np.full(len(lb), np.inf)), method="bvls").x
        except Exception:
            xs = np.zeros(A.shape[1])
        res = g - A @ xs
        if not np.all(np.isfinite(xs)) or np.linalg.norm(res) > np.linalg.norm(g):
            a = F["aeq"]; xs = np.zeros(A.shape[1]); xs[0] = (a @ g) / (a @ a); res = g - A @ xs
        return xs[0], (xs[1] if vol_active else 0.0), float(np.linalg.norm(res) / np.linalg.norm(g))

    def subproblem(self, F, p, H, Delta):
        P = self.P
        lo = np.array([max(b[0] - pi, -Delta) for b, pi in zip(P.bounds(), p)])
        hi = np.array([min(b[1] - pi, Delta) for b, pi in zip(P.bounds(), p)])
        cons = [dict(type="eq", fun=lambda d: F["ceq"] + F["aeq"] @ d, jac=lambda d: F["aeq"]),
                dict(type="ineq", fun=lambda d: F["cv"] + F["av"] @ d, jac=lambda d: F["av"])]
        if len(P.Lsm):
            sm0 = P.c_smooth(p)
            cons.append(dict(type="ineq", fun=lambda d: sm0 + self.Jsm @ d, jac=lambda d: self.Jsm))
        res = minimize(lambda d: F["g"] @ d + 0.5 * d @ H @ d, np.zeros(P.np), jac=lambda d: F["g"] + H @ d,
                       method="SLSQP", bounds=list(zip(lo, hi)), constraints=cons,
                       options=dict(maxiter=500, ftol=1e-12))
        d = res.x
        # second-order correction for the volume: smallest shift d += A'y that brings V back to its bound
        # without changing the linearised CMy
        A = np.vstack([F["av"], F["aeq"]])
        for _ in range(4):
            cvn = P.c_vol(p + d)
            if cvn >= 1e-4:
                break
            d = d + A.T @ np.linalg.solve(A @ A.T, np.array([-cvn + 2e-4, 0.0]))
        pred = -(F["g"] @ d + 0.5 * d @ H @ d) + self.nu * (abs(F["ceq"]) - abs(F["ceq"] + F["aeq"] @ d)) \
            + self.nu * (max(0, -F["cv"]) - max(0, -(F["cv"] + F["av"] @ d)))
        return d, pred

    def feasible(self, F):
        r = F["r"]
        return abs(r["CMy"]) < 2e-5 and F["cv"] > -TOL_ACTIVE and r["smooth"] < 1e-4

    def run(self, maxiter=60, delta0=0.15, budget_s=None, stop_dk=1e-4, kkt_tol=1e-2):
        P, p = self.P, self.p
        log = os.path.join(P.runs, "trsqp_log.csv")
        its = os.path.join(P.runs, "opt_iterations.txt")
        if not os.path.exists(log):
            open(log, "w").write("k,eval,accepted,K,CD,CMy,Vrel,Delta,rho,kkt_rel,step_inf,time_s\n")
        F = self.funcs(p)
        Delta = delta0
        H = np.eye(P.np) * max(np.abs(F["g"]).max() / 0.3, 1e-3)
        Ks = [F["r"]["CL"] / F["r"]["CD"]]
        lam, mu, kkt = self.multipliers(F, p)
        with open(its, "a") as fh:
            fh.write(f"trsqp start from eval {F['r']['n']}  kkt={kkt:.3e}\n")
        stop = "maxiter"
        for k in range(maxiter):
            if budget_s and time.time() - self.t0 > budget_s:
                stop = "time budget"; break
            d, pred = self.subproblem(F, p, H, Delta)
            pn = p + d
            Fn = self.funcs(pn, grad=False)
            rho = (self.merit(F) - self.merit(Fn)) / pred if pred > 1e-12 else -1.0
            acc = rho > 0.05
            if acc:
                Fn = self.funcs(pn, grad=True)
                lam_n, mu_n, kkt_n = self.multipliers(Fn, pn)
                gl = lambda FF: FF["g"] - lam_n * FF["aeq"] - mu_n * FF["av"]   # noqa: E731
                s, y = d, gl(Fn) - gl(F)
                sHs, sy = s @ H @ s, s @ y
                if sy < 0.2 * sHs:                        # Powell damping keeps H positive definite
                    th = 0.8 * sHs / (sHs - sy); y = th * y + (1 - th) * H @ s; sy = s @ y
                if sy > 1e-12 and sHs > 1e-12:
                    Hs = H @ s
                    H = H - np.outer(Hs, Hs) / sHs + np.outer(y, y) / sy
                p, F, lam, mu, kkt = pn, Fn, lam_n, mu_n, kkt_n
                r = F["r"]
                Ks.append(r["CL"] / r["CD"])
                with open(its, "a") as fh:
                    fh.write(f"iter T{k + 1} -> eval {r['n']}  CD={r['CD']:.7f} CMy={r['CMy']:+.7f} K={Ks[-1]:.4f} "
                             f"V={r['Vrel']:.5f} smooth={r['smooth']:+.4f} kkt={kkt:.3e} Delta={Delta:.3f}\n")
            if rho > 0.75 and np.abs(d).max() > 0.9 * Delta:
                Delta = min(2 * Delta, 1.0)
            elif rho < 0.25:
                Delta = 0.5 * Delta if acc else max(0.3 * Delta, 1e-3)
            rn = Fn["r"]
            with open(log, "a") as fh:
                fh.write(f"{k + 1},{rn['n']},{int(acc)},{rn['CL'] / rn['CD']:.5f},{rn['CD']:.8f},{rn['CMy']:.8f},"
                         f"{rn['Vrel']:.6f},{Delta:.4f},{rho:.3f},{kkt:.4e},{np.abs(d).max():.4f},"
                         f"{time.time() - self.t0:.0f}\n")
            print(f"k={k + 1} accepted={acc} rho={rho:.2f} Delta={Delta:.3f} K={Ks[-1]:.4f} "
                  f"CMy={F['r']['CMy']:+.2e} kkt={kkt:.2e}", flush=True)
            if acc and self.feasible(F):
                if len(Ks) >= 4 and max(abs(Ks[-m] / Ks[-m - 1] - 1) for m in (1, 2, 3)) < stop_dk:
                    stop = f"relative change of K < {stop_dk:g} on three steps"; break
                if kkt < kkt_tol:
                    stop = f"KKT residual < {kkt_tol:g}"; break
            if Delta < 2e-3:
                stop = "trust radius < 2 mm"; break
        out = dict(stop=stop, final_eval=F["r"]["n"], K=Ks[-1], CD=F["r"]["CD"], CMy=F["r"]["CMy"],
                   V_rel=F["r"]["Vrel"], kkt_rel=kkt, elapsed_s=time.time() - self.t0)
        json.dump(out, open(os.path.join(P.runs, "trsqp_result.json"), "w"), indent=1)
        print(json.dumps(out, indent=1), flush=True)
        return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", default="runs/wing_body")
    ap.add_argument("--start-eval", type=int, default=None,
                    help="design to start from (default: final_eval of opt_result.json written by driver.py)")
    ap.add_argument("--maxiter", type=int, default=60)
    ap.add_argument("--delta0", type=float, default=0.15, help="initial trust radius, m")
    ap.add_argument("--nu", type=float, default=20.0, help="penalty weight in the merit function")
    ap.add_argument("--budget-hours", type=float, default=None)
    ap.add_argument("--stop-dk", type=float, default=1e-4)
    ap.add_argument("--kkt-tol", type=float, default=1e-2)
    a = ap.parse_args()
    S.check_su2()
    P = Problem.load(a.workdir)
    start = a.start_eval
    if start is None:
        start = json.load(open(os.path.join(P.runs, "opt_result.json")))["final_eval"]
    TRSQP(P, start, nu=a.nu).run(maxiter=a.maxiter, delta0=a.delta0,
                                 budget_s=a.budget_hours * 3600 if a.budget_hours else None,
                                 stop_dk=a.stop_dk, kkt_tol=a.kkt_tol)


if __name__ == "__main__":
    main()
