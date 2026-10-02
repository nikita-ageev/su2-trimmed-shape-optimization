"""One command per run of the wing-body case (see scripts/presets.py for what a preset changes).

    python run.py --preset robust             # mesh (if missing) -> baseline + adjoints -> trust-region SQP
    python run.py --preset verify             # 2nd-order (JST) check of the robust baseline and final design
    python run.py --preset v02                # the v0.2 setup (JST, 15 x 5 x 5 FFD, no clamps), for comparison

Options: --workdir (default runs/<preset>; verify reads runs/robust and writes runs/robust_verify), --mesh-h
(surface mesh size, default 0.25 m; 0.6 m gives a ~20-minute test), --maxiter, --budget-hours, --np (MPI ranks,
also SU2_NP). Results: <workdir>/trsqp_result.json, smooth.json (nose smoothness, scripts/smoothness.py),
verify.json; one folder per design (dsn_NNN/).
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import presets as PR  # noqa: E402
import geometry as G  # noqa: E402

ZONE = (0.01 * G.L_FUS, G.X_LE_ROOT - 0.01 * G.L_FUS)    # nose zone: the fuselage ahead of the wing root


def smooth(P, base_dir, final_dir, h):
    import smoothness as SM
    m = SM.evaluate(os.path.join(P.setup, "mesh_ffd.su2"), base_dir, final_dir, ZONE, h, G.L_FUS)
    json.dump(m, open(os.path.join(P.runs, "smooth.json"), "w"), indent=1)
    print("smoothness base ", SM.totals(m, "base"), "\nsmoothness final", SM.totals(m, "final"), flush=True)
    return m


def run_optimisation(a, preset):
    import su2run as S
    from driver import Problem, Evaluator
    import trsqp
    S.check_su2()
    workdir = a.workdir or os.path.join(ROOT, "runs", preset)
    mesh = os.path.join(ROOT, "mesh", f"wing_body_h{a.mesh_h:g}.su2")
    if not os.path.exists(os.path.join(workdir, "setup", "mesh_ffd.su2")) and not os.path.exists(mesh):
        os.makedirs(os.path.dirname(mesh), exist_ok=True)
        subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "make_mesh.py"), mesh, str(a.mesh_h)], check=True)
    P = Problem(workdir, mesh=mesh, **PR.problem_kwargs(preset))
    print(f"preset {preset}: {P.np} variables, numerics {P.settings['numerics']}, clamped planes "
          f"{P.settings['clamp_i']}, Sobolev {P.settings['sobolev']}", flush=True)
    E = Evaluator(P)
    if not os.path.exists(os.path.join(P.runs, "dsn_000", "grad_CD.txt")):
        E.gradients(np.zeros(P.np))
    out = trsqp.TRSQP(P, 0).run(maxiter=a.maxiter, delta0=0.15,
                                budget_s=a.budget_hours * 3600 if a.budget_hours else None)
    smooth(P, os.path.join(P.runs, "dsn_000"), os.path.join(P.runs, f"dsn_{out['final_eval']:03d}"), a.mesh_h)
    return out


def run_verify(a):
    import su2run as S
    from driver import Problem, Evaluator
    S.check_su2()
    src = a.workdir or os.path.join(ROOT, "runs", "robust")
    fin = json.load(open(os.path.join(src, "trsqp_result.json")))["final_eval"]
    st = json.load(open(os.path.join(src, "setup", "settings.json")))
    st.update(PR.problem_kwargs("verify"))
    work = src.rstrip("/") + "_verify"
    os.makedirs(os.path.join(work, "setup"), exist_ok=True)
    link = os.path.join(work, "setup", "mesh_ffd.su2")
    if not os.path.exists(link):
        os.symlink(os.path.join(os.path.abspath(src), "setup", "mesh_ffd.su2"), link)
    P = Problem(work, **st)
    E = Evaluator(P, prefix="ver")
    p_fin = np.loadtxt(os.path.join(src, f"dsn_{fin:03d}", "p.txt"))
    r0, r1 = E.primal(np.zeros(P.np)), E.primal(p_fin)
    out = {tag: dict(CD=r["CD"], CL=r["CL"], CMy=r["CMy"], AoA=r["AoA"], K=r["CL"] / r["CD"])
           for tag, r in (("base", r0), ("final", r1))}
    out.update(final_eval_robust=fin, dK_pct=100 * (out["final"]["K"] / out["base"]["K"] - 1))
    json.dump(out, open(os.path.join(work, "verify.json"), "w"), indent=1)
    print(json.dumps(out, indent=1), flush=True)
    smooth(P, r0["dir"], r1["dir"], a.mesh_h)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=sorted(PR.PRESETS), default="robust")
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--mesh-h", type=float, default=0.25)
    ap.add_argument("--maxiter", type=int, default=30)
    ap.add_argument("--budget-hours", type=float, default=None)
    ap.add_argument("--np", type=int, default=None, help="MPI ranks (default: $SU2_NP or 8)")
    a = ap.parse_args()
    if a.np:
        os.environ["SU2_NP"] = str(a.np)
    if a.preset == "verify":
        run_verify(a)
    else:
        run_optimisation(a, a.preset)


if __name__ == "__main__":
    main()
