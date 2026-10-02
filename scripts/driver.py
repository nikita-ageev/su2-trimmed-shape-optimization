"""Trimmed aerodynamic shape optimisation with SU2 (discrete adjoint) and SciPy SLSQP.

Problem (wing-body half model, Euler, M = 1.7):

    max  K = CL / CD                          (CL is held by SU2, so this is  min CD)
    s.t. CL(p)  = CL*                         lift preserved: FIXED_CL_MODE, SU2 trims the angle of attack
         CMy(p) = 0                           pitching moment about the centre of gravity: equality constraint
         V(p)  >= f_V * V0                    fuselage volume (default f_V = 0.995)
         L(p)   = L0                          length fixed: control points move only in y and z
         |p_i| <= bounds,  [optional] |second differences of the control-point displacements| <= s

Each design evaluation: SU2_DEF (FFD mesh deformation) -> SU2_CFD (fixed CL); the gradient needs one fixed-AoA
run at AoA + 0.1 deg (dCD/dCL, dCMy/dCL), two discrete adjoints (SU2_CFD_AD for CD and for CMy, both at constant
CL) and SU2_DOT_AD (projection onto the FFD variables). The volume and its gradient are computed here from the
surface mesh (see su2run.Volume).

Usage (from the repository root):
    python scripts/make_mesh.py mesh/wing_body.su2 0.25
    SU2_NP=8 python scripts/driver.py --mesh mesh/wing_body.su2 --workdir runs/wing_body --maxiter 15
Use a fresh --workdir for every optimisation (the FFD-embedded mesh in <workdir>/setup is reused).
"""
import argparse
import json
import os
import shutil
import sys
import time

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import su2run as S  # noqa: E402
import geometry as G  # noqa: E402
import presets as PR  # noqa: E402

F_SCALE, C_SCALE = 1e4, 1e3        # objective CD in drag counts, CMy in 1e-3
PRESET_OPTIONS = {
    "bezier": dict(bnd_z=1.0, bnd_y=0.5, link_z01=False, smooth=None),
    "bspline": dict(bnd_z=1.5, bnd_y=0.8, link_z01=True, smooth=(0.4, 0.25)),
    # 9 planes along x instead of 15: the same curvature limit needs (8/14)^2 of the second difference; tightened
    # further by 2 (see scripts/presets.py)
    "bspline_low": dict(bnd_z=1.5, bnd_y=0.8, link_z01=True, smooth=(0.2, 0.125)),
}


class Stop(Exception):
    pass


class Problem:
    """Design space, SU2 config generation and constraint functions. Settings are stored in
    <workdir>/setup/settings.json so that fd_check.py, kkt.py and post.py can rebuild the same problem."""

    def __init__(self, workdir, mesh=None, template=None, ffd="bezier", target_cl=0.10, xcg=G.X_CG,
                 vol_frac=0.995, aoa0=2.66, bnd_z=None, bnd_y=None, link_z01=None, smooth=None,
                 direct_minval=-10.5, adj_minval=-8.5, direct_iter=3000, adj_iter=2500, dcx="alpha",
                 numerics="second_order", clamp_i=(), sobolev=None, preset=None, **_):
        opt = PRESET_OPTIONS[ffd]
        self.settings = dict(ffd=ffd, target_cl=target_cl, xcg=xcg, vol_frac=vol_frac, aoa0=aoa0,
                             bnd_z=opt["bnd_z"] if bnd_z is None else bnd_z,
                             bnd_y=opt["bnd_y"] if bnd_y is None else bnd_y,
                             link_z01=opt["link_z01"] if link_z01 is None else link_z01,
                             smooth=opt["smooth"] if smooth is None else (None if smooth == "none" else smooth),
                             template=os.path.abspath(template or os.path.join(S.REPO, "config", "wing_body.cfg")),
                             direct_minval=direct_minval, adj_minval=adj_minval,
                             direct_iter=direct_iter, adj_iter=adj_iter, dcx=dcx,
                             numerics=numerics, clamp_i=sorted(int(i) for i in clamp_i),
                             sobolev=list(sobolev) if sobolev else None, preset=preset)
        s = self.settings
        self.runs = os.path.abspath(workdir)
        self.setup = os.path.join(self.runs, "setup")
        os.makedirs(self.setup, exist_ok=True)
        S.TIMING_CSV = os.path.join(self.runs, "timing.csv")
        self.spec = S.FFDSpec(ffd, G.FFD_PRESETS[ffd])
        self.ffd_mesh = os.path.join(self.setup, "mesh_ffd.su2")
        if not os.path.exists(self.ffd_mesh):
            if mesh is None:
                raise SystemExit("--mesh is required for a new workdir")
            shutil.copy(mesh, os.path.join(self.setup, "mesh.su2"))     # config paths must not contain spaces
            self.cfg(os.path.join(self.setup, "ffd.cfg"), None, MESH_IN="mesh.su2", MESH_OUT="mesh_ffd.su2",
                     DV_KIND="FFD_SETTING", DV_PARAM="( FUS, 1.0 )", DV_VALUE="0.0")
            S.run("SU2_DEF", "ffd.cfg", self.setup, "log_ffd.txt")
            os.remove(os.path.join(self.setup, "mesh.su2"))
        json.dump(s, open(os.path.join(self.setup, "settings.json"), "w"), indent=1)
        self._build_variables()
        self.vol = S.Volume(self.ffd_mesh, self.spec)
        self.V0 = G.V_FUS_HALF   # everything outside the FFD box is fixed -> volume change = fuselage volume change

    @classmethod
    def load(cls, workdir):
        s = json.load(open(os.path.join(workdir, "setup", "settings.json")))
        if s.get("smooth") is None:
            s["smooth"] = "none"
        return cls(workdir, **s)

    # ---------------------------------------------------------------- design variables
    def _build_variables(self):
        s, spec = self.settings, self.spec
        L, M, N = spec.box["deg"]
        J = spec.box["fixj"]
        jz = range(1, J) if s["link_z01"] else range(J)     # link j = 0 and j = 1 -> smooth crest/keel at y = 0
        free_i = [i for i in range(L + 1) if i not in set(s.get("clamp_i") or ())]   # clamped planes: no variables
        self.pvars = [("z", i, j, k) for i in free_i for j in jz for k in range(N + 1)] + \
                     [("y", i, j, k) for i in free_i for j in range(1, J) for k in range(N + 1)]
        self.np = len(self.pvars)
        pidx = {v: n for n, v in enumerate(self.pvars)}
        self.T = np.zeros((spec.ndv, self.np))     # SU2 design variables x = T p
        for n, (i, j, k, d) in enumerate(spec.dvs):
            if i not in free_i:
                continue                                   # SU2 variable kept at 0
            if d == 2:
                self.T[n, pidx[("z", i, max(j, 1) if s["link_z01"] else j, k)]] = 1.0
            else:
                self.T[n, pidx[("y", i, j, k)]] = 1.0
        # linear smoothness constraints on second differences of the displacement field
        self.Dsm = np.zeros((0, self.np))
        self.Lsm = np.zeros(0)
        if s["smooth"]:
            sx, syz = s["smooth"]
            row = {dv: n for n, dv in enumerate(spec.dvs)}
            rows, lims = [], []
            for dirn, d in (("z", 2), ("y", 1)):
                A = np.zeros((L + 1, J + 1, N + 1, self.np))
                for i in range(L + 1):
                    for j in range(J):
                        for k in range(N + 1):
                            if (i, j, k, d) in row:
                                A[i, j, k] = self.T[row[(i, j, k, d)]]
                for ax, lim in ((0, sx), (1, syz), (2, syz)):
                    for m in range(1, A.shape[ax] - 1):
                        D = (np.take(A, m - 1, axis=ax) - 2 * np.take(A, m, axis=ax) +
                             np.take(A, m + 1, axis=ax)).reshape(-1, self.np)
                        for r in D:
                            if np.any(r != 0):
                                rows.append(r); lims.append(lim)
            U = np.unique(np.c_[np.array(rows), np.array(lims)], axis=0)
            self.Dsm, self.Lsm = U[:, :-1], U[:, -1]

    def metric(self):
        """Sobolev metric of the design space, M = I + eps_x Dx'Dx + eps_yz Dyz'Dyz (D: second differences of the
        control-point displacements along x and across the box). trsqp.py uses it as the initial Hessian, so a
        zig-zag of the control net (period of two planes) costs ~1 + 16 eps and a smooth bend ~1. Identity when
        the preset has no smoothing (v0.2 behaviour)."""
        return PR.sobolev_metric(self.pvars, self.settings.get("sobolev"))

    def bounds(self):
        s = self.settings
        return [(-s["bnd_z"], s["bnd_z"]) if v[0] == "z" else (-s["bnd_y"], s["bnd_y"]) for v in self.pvars]

    # ---------------------------------------------------------------- SU2 configs
    def cfg(self, path, x, **kw):
        s = self.settings
        d = dict(MATH_PROBLEM="DIRECT", RESTART_SOL="NO", AOA="2.0", DCD_DCL="0.0", DCMY_DCL="0.0",
                 FIXED_CL="YES", EVAL_DOF_DCX="YES" if s.get("dcx") == "su2" else "NO", DISCARD_INFILES="NO",
                 XCG=f"{s['xcg']}", TARGET_CL=f"{s['target_cl']}", OBJ="DRAG", ITER=str(s["direct_iter"]),
                 CONV_FIELD="RMS_DENSITY", MINVAL=str(s["direct_minval"]), MESH_IN="mesh.su2", MESH_OUT="mesh_out.su2",
                 SOLUTION="solution_flow.dat", RESTART="restart_flow.dat", OUTPUT_FILES="(RESTART, SURFACE_CSV)",
                 CONV_FILENAME="history", SURFACE_FILENAME="surface_flow", SURFACE_ADJ_FILENAME="surface_adjoint",
                 GRAD_FILENAME="of_grad.csv",
                 SCREEN_OUTPUT="(INNER_ITER, RMS_DENSITY, RMS_ENERGY, LIFT, DRAG, MOMENT_Y, AOA)",
                 HISTORY_OUTPUT="(ITER, RMS_RES, AERO_COEFF, AOA)",
                 NUMERICS=PR.NUMERICS[s.get("numerics", "second_order")])
        d.update(self.spec.cfg_fields())
        d.update(self.spec.dv_strings(np.zeros(self.spec.ndv) if x is None else x))
        d.update({k: str(v) for k, v in kw.items()})
        S.fill_template(s["template"], d, path)

    # ---------------------------------------------------------------- constraints (scaled for SLSQP)
    def v_rel(self, p):
        return (self.V0 + self.vol.value(self.T @ p) - self.vol.V0) / self.V0

    def c_vol(self, p):
        return 100.0 * (self.v_rel(p) - self.settings["vol_frac"])

    def dc_vol(self, p):
        return 100.0 * (self.T.T @ self.vol.grad(self.T @ p)) / self.V0

    def c_smooth(self, p):
        return 10.0 * np.r_[self.Lsm - self.Dsm @ p, self.Lsm + self.Dsm @ p]

    def dc_smooth(self, p):
        return 10.0 * np.r_[-self.Dsm, self.Dsm]

    def smooth_violation(self, p):
        return float(np.max(np.abs(self.Dsm @ p) - self.Lsm)) if len(self.Lsm) else 0.0


class Evaluator:
    """Caches primal and adjoint results per design; each design gets its own folder <prefix>_NNN."""

    def __init__(self, P, prefix="dsn", t0=None):
        self.P, self.prefix, self.cache, self.n = P, prefix, [], 0
        self.t0 = t0 or time.time()
        self.last_restart, self.last_aoa = None, P.settings["aoa0"]
        self.log = os.path.join(P.runs, f"{prefix}_history.csv")
        if not os.path.exists(self.log):
            open(self.log, "w").write("eval,time_s,CD,CL,CMy,AoA,K,Vfus_rel,rho0,rho_end,iters,has_grad,"
                                      "max_abs_dv,smooth_viol\n")

    def find(self, p):
        for c in self.cache:
            if np.allclose(c["p"], p, rtol=0, atol=1e-12):
                return c
        return None

    def load_eval(self, ev):
        """Put a design computed earlier (folder <prefix>_NNN, possibly by another process) into the cache
        without recomputing it; the next new design restarts from its flow solution."""
        P = self.P
        d = os.path.join(P.runs, f"{self.prefix}_{ev:03d}")
        p = np.loadtxt(os.path.join(d, "p.txt"))
        r = S.direct_summary(d)
        mesh = "mesh_def.su2" if os.path.exists(os.path.join(d, "mesh_def.su2")) else "../setup/mesh_ffd.su2"
        r.update(p=p, x=P.T @ p, dir=d, mesh=mesh, n=ev, Vrel=P.v_rel(p), smooth=P.smooth_violation(p))
        if os.path.exists(os.path.join(d, "dcx.json")):
            r.update(json.load(open(os.path.join(d, "dcx.json"))))
        if os.path.exists(os.path.join(d, "grad_CD.txt")) and (P.settings.get("dcx", "alpha") != "alpha"
                                                              or os.path.exists(os.path.join(d, "dcx.json"))):
            r["gCD"] = np.loadtxt(os.path.join(d, "grad_CD.txt"))
            r["gCMy"] = np.loadtxt(os.path.join(d, "grad_CMy.txt"))
        self.cache.append(r)
        self.last_restart, self.last_aoa = f"../{self.prefix}_{ev:03d}/restart_flow.dat", r["AoA"]
        return r

    def restore(self):
        """Reload every completed design folder of the workdir (used by drivers that run one evaluation per
        process, such as dakota_driver.py). Incomplete folders (no flow.meta) are skipped."""
        runs, pre = self.P.runs, self.prefix + "_"
        evs = sorted(int(n[len(pre):]) for n in os.listdir(runs) if n.startswith(pre) and n[len(pre):].isdigit())
        for ev in evs:
            if os.path.exists(os.path.join(runs, f"{pre}{ev:03d}", "flow.meta")):
                self.load_eval(ev)
        self.n = evs[-1] + 1 if evs else 0
        return self

    def primal(self, p):
        P = self.P
        p = np.array(p, dtype=float, copy=True)        # SLSQP modifies its array in place: keep a copy
        c = self.find(p)
        if c is not None:
            return c
        x = P.T @ p
        d = os.path.join(P.runs, f"{self.prefix}_{self.n:03d}")
        os.makedirs(d, exist_ok=True)
        np.savetxt(os.path.join(d, "p.txt"), p)
        np.savetxt(os.path.join(d, "x.txt"), x)
        if np.any(x != 0):
            P.cfg(os.path.join(d, "def.cfg"), x, MESH_IN="../setup/mesh_ffd.su2", MESH_OUT="mesh_def.su2")
            S.run("SU2_DEF", "def.cfg", d, "log_def.txt")
            mesh = "mesh_def.su2"
        else:
            mesh = "../setup/mesh_ffd.su2"
        rst = self.last_restart
        P.cfg(os.path.join(d, "direct.cfg"), x, MESH_IN=mesh, AOA=f"{self.last_aoa:.10f}",
              RESTART_SOL="YES" if rst else "NO", SOLUTION=rst or "solution_flow.dat",
              OUTPUT_FILES="(RESTART, SURFACE_CSV, PARAVIEW)")
        S.run("SU2_CFD", "direct.cfg", d, "log_direct.txt")
        r = S.direct_summary(d)
        self.last_restart = f"../{os.path.basename(d)}/restart_flow.dat"
        self.last_aoa = r["AoA"]
        r.update(p=p, x=x, dir=d, mesh=mesh, n=self.n, Vrel=P.v_rel(p), smooth=P.smooth_violation(p))
        self.cache.append(r)
        self.n += 1
        K = r["CL"] / r["CD"]
        with open(self.log, "a") as f:
            f.write(f"{r['n']},{time.time() - self.t0:.0f},{r['CD']:.8f},{r['CL']:.6f},{r['CMy']:.8f},{r['AoA']:.5f},"
                    f"{K:.5f},{r['Vrel']:.6f},{r['rho0']:.3f},{r['rho_end']:.3f},{r['iters']},0,"
                    f"{np.abs(p).max():.4f},{r['smooth']:.4f}\n")
        print(f"[{self.prefix} {r['n']:3d}] CD={r['CD']:.6f} CL={r['CL']:.4f} CMy={r['CMy']:+.6f} K={K:.3f} "
              f"V/V0={r['Vrel']:.5f} AoA={r['AoA']:.3f} rho={r['rho_end']:.1f} t={time.time() - self.t0:.0f}s",
              flush=True)
        return r

    def dcx(self, r, dalpha=0.1):
        """dCD/dCL and dCMy/dCL by a one-sided difference: a fixed-AoA run at AoA + dalpha restarted from the
        converged fixed-CL solution. SU2's own estimate (EVAL_DOF_DCX= YES) can be wrong after a restart: when the
        ITER_DCL_DALPHA stage ends on an iteration that is not written, the derivatives in flow.meta are not
        updated (seen here as dCMy/dCL jumping from -0.04 to +0.85). Results are stored in dcx.json."""
        P, d = self.P, r["dir"]
        f = os.path.join(d, "dcx.json")
        if os.path.exists(f):
            r.update(json.load(open(f)))
            return
        P.cfg(os.path.join(d, "dalpha.cfg"), r["x"], MESH_IN=r["mesh"], AOA=f"{r['AoA'] + dalpha:.10f}",
              FIXED_CL="NO", DISCARD_INFILES="YES", EVAL_DOF_DCX="NO", RESTART_SOL="YES",
              SOLUTION="restart_flow.dat", RESTART="restart_da.dat", CONV_FILENAME="history_da",
              MINVAL=str(P.settings["direct_minval"] + 1.0), OUTPUT_FILES="(RESTART)")
        S.run("SU2_CFD", "dalpha.cfg", d, "log_dalpha.txt")
        hdr, a = S.read_history(os.path.join(d, "history_da.csv"))
        last = dict(zip(hdr, a[-1]))
        dCL = last["CL"] - r["CL"]
        out = dict(dCD_dCL=(last["CD"] - r["CD"]) / dCL, dCMy_dCL=(last["CMy"] - r["CMy"]) / dCL,
                   dCL_dalpha=dCL / dalpha)
        os.remove(os.path.join(d, "restart_da.dat"))
        json.dump(out, open(f, "w"), indent=1)
        r.update(out)

    def gradients(self, p):
        """Gradients of CD and CMy at constant CL w.r.t. the optimiser variables p (two discrete adjoints)."""
        P = self.P
        r = self.primal(p)
        if "gCD" in r:
            return r
        d = r["dir"]
        if P.settings.get("dcx", "alpha") == "alpha":
            self.dcx(r)
        for obj, tag, key in (("DRAG", "cd", "CD"), ("MOMENT_Y", "cmy", "CMy")):
            cfg = f"adj_{tag}.cfg"
            P.cfg(os.path.join(d, cfg), r["x"], MESH_IN=r["mesh"], MATH_PROBLEM="DISCRETE_ADJOINT", OBJ=obj,
                  SOLUTION="restart_flow.dat", AOA=f"{r['AoA']:.12f}", DCD_DCL=f"{r['dCD_dCL']:.12g}",
                  DCMY_DCL=f"{r['dCMy_dCL']:.12g}", CONV_FIELD="RMS_ADJ_DENSITY",
                  # SU2 silently resets DISCARD_INFILES to NO for a fixed-CL adjoint unless EVAL_DOF_DCX= YES
                  # (CConfig.cpp); with NO it would take dJ/dCL from flow.meta instead of the values above
                  DISCARD_INFILES="YES", EVAL_DOF_DCX="YES",
                  MINVAL=str(P.settings["adj_minval"]), ITER=str(P.settings["adj_iter"]),
                  CONV_FILENAME=f"history_adj_{tag}", SURFACE_FILENAME=f"surface_flow_adj_{tag}",
                  SURFACE_ADJ_FILENAME=f"surface_adjoint_{tag}", GRAD_FILENAME=f"of_grad_{tag}.csv",
                  SCREEN_OUTPUT="(INNER_ITER, RMS_ADJ_DENSITY, RMS_ADJ_ENERGY, SENS_AOA)",
                  HISTORY_OUTPUT="(ITER, RMS_RES, SENSITIVITY)")
            S.run("SU2_CFD_AD", cfg, d, f"log_adj_{tag}.txt")
            # SU2_DOT_AD reads the adjoint solution from SOLUTION_ADJ_FILENAME (+ objective suffix)
            shutil.copy(os.path.join(d, f"restart_adj_{tag}.dat"), os.path.join(d, f"solution_adj_{tag}.dat"))
            S.run("SU2_DOT_AD", cfg, d, f"log_dot_{tag}.txt")
            os.remove(os.path.join(d, f"solution_adj_{tag}.dat"))
            gx = S.read_grad(os.path.join(d, f"of_grad_{tag}.csv"))
            r["gx" + key] = gx
            r["g" + key] = P.T.T @ gx
            np.savetxt(os.path.join(d, f"grad_{key}.txt"), r["g" + key])
            np.savetxt(os.path.join(d, f"gradx_{key}.txt"), gx)
        lines = open(self.log).readlines()
        with open(self.log, "w") as f:
            for line in lines:
                parts = line.split(",")
                if parts[0] == str(r["n"]):
                    parts[11] = "1"
                    line = ",".join(parts)
                f.write(line)
        return r


def optimise(P, maxiter=15, budget_s=None, ftol=1e-9, stop_dk=1e-4, cmy_tol=2e-5):
    E = Evaluator(P)
    iters = []
    t0 = E.t0

    def f(p):
        if budget_s and time.time() - t0 > budget_s:
            raise Stop("time budget exhausted")
        return F_SCALE * E.primal(p)["CD"]

    def df(p):
        return F_SCALE * E.gradients(p)["gCD"]

    def c_eq(p):
        return C_SCALE * E.primal(p)["CMy"]

    def dc_eq(p):
        return C_SCALE * E.gradients(p)["gCMy"]

    def feasible(r):
        return (abs(r["CMy"]) < cmy_tol and r["Vrel"] >= P.settings["vol_frac"] - 2e-5 and r["smooth"] < 1e-4)

    def callback(pk):
        r = E.primal(pk)
        iters.append(r["n"])
        K = r["CL"] / r["CD"]
        with open(os.path.join(P.runs, "opt_iterations.txt"), "a") as fh:
            fh.write(f"iter {len(iters)} -> eval {r['n']}  CD={r['CD']:.7f} CMy={r['CMy']:+.7f} K={K:.4f} "
                     f"V={r['Vrel']:.5f} smooth={r['smooth']:+.4f}\n")
        Ks = [E.cache[n]["CL"] / E.cache[n]["CD"] for n in iters]
        if len(Ks) >= 4 and feasible(r):
            if max(abs(Ks[-m] / Ks[-m - 1] - 1) for m in (1, 2, 3)) < stop_dk:
                raise Stop(f"relative change of K < {stop_dk:g} on three iterations")

    cons = [dict(type="eq", fun=c_eq, jac=dc_eq), dict(type="ineq", fun=P.c_vol, jac=P.dc_vol)]
    if len(P.Lsm):
        cons.append(dict(type="ineq", fun=P.c_smooth, jac=P.dc_smooth))
    p0 = np.zeros(P.np)
    print(f"SU2 design variables: {P.spec.ndv}, optimiser variables: {P.np}, smoothness rows: {len(P.Lsm)}",
          flush=True)
    try:
        res = minimize(f, p0, jac=df, method="SLSQP", bounds=P.bounds(), constraints=cons, callback=callback,
                       options=dict(maxiter=maxiter, ftol=ftol, disp=True))
        msg = res.message
    except Stop as e:
        msg = str(e)
    base = E.cache[0]
    last = E.cache[iters[-1]] if iters else base
    feas = [c for c in E.cache if feasible(c)]
    best = min(feas, key=lambda c: c["CD"]) if feas else None
    K0, K1 = base["CL"] / base["CD"], last["CL"] / last["CD"]
    out = dict(message=str(msg), n_evals=E.n, iterations=iters, final_eval=last["n"],
               best_feasible_eval=best["n"] if best else None, elapsed_s=time.time() - t0,
               K0=K0, K_final=K1, dK_pct=100 * (K1 / K0 - 1), CD0=base["CD"], CD_final=last["CD"],
               CMy0=base["CMy"], CMy_final=last["CMy"], V_rel_final=last["Vrel"], AoA0=base["AoA"], AoA_final=last["AoA"])
    json.dump(out, open(os.path.join(P.runs, "opt_result.json"), "w"), indent=1)
    print(json.dumps(out, indent=1), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mesh", help="input SU2 mesh (markers aircraft, symmetry, farfield)")
    ap.add_argument("--workdir", default="runs/wing_body")
    ap.add_argument("--template", default=None, help="SU2 config template (default config/wing_body.cfg)")
    ap.add_argument("--ffd", choices=sorted(G.FFD_PRESETS), default="bezier")
    ap.add_argument("--maxiter", type=int, default=15)
    ap.add_argument("--target-cl", type=float, default=0.10)
    ap.add_argument("--xcg", type=float, default=G.X_CG, help="moment reference (centre of gravity), m")
    ap.add_argument("--vol-frac", type=float, default=0.995, help="V >= vol_frac * V0")
    ap.add_argument("--bnd-z", type=float, default=None)
    ap.add_argument("--bnd-y", type=float, default=None)
    ap.add_argument("--budget-hours", type=float, default=None)
    ap.add_argument("--stop-dk", type=float, default=1e-4)
    ap.add_argument("--dcx", choices=("alpha", "su2"), default="alpha",
                    help="dCD/dCL, dCMy/dCL for the constant-CL gradient: separate run at AoA + 0.1 deg (default) "
                         "or SU2's EVAL_DOF_DCX estimate from flow.meta")
    a = ap.parse_args()
    S.check_su2()
    P = Problem(a.workdir, mesh=a.mesh, template=a.template, ffd=a.ffd, target_cl=a.target_cl, xcg=a.xcg,
                vol_frac=a.vol_frac, bnd_z=a.bnd_z, bnd_y=a.bnd_y, dcx=a.dcx)
    optimise(P, maxiter=a.maxiter, budget_s=a.budget_hours * 3600 if a.budget_hours else None, stop_dk=a.stop_dk)


if __name__ == "__main__":
    main()
