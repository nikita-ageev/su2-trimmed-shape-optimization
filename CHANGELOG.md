# Changelog

## 0.3.0 (2026-10-02)

Robust preset: no surface waves, no bumps on the nose.

* Diagnosis of v0.2 run 2 (450 variables): the second differences of the final control net change sign 5.2 times
  out of 12 possible along an average control line (a zig-zag with a period of two control planes), the free nose
  planes moved by up to 0.82 m, and the nose zone of the fuselage got 10 new waves of the wall displacement and two
  bumps of up to 139 mm on a body that had none (`scripts/smoothness.py`). Cause: free nose planes, which carry the
  largest pressure gradients; an identity initial Hessian in `trsqp.py`, so the first steps follow the raw,
  plane-to-plane alternating adjoint gradient; loose second-difference limits.
* `scripts/presets.py` and `run.py`: one command per run, `python run.py --preset robust|verify|v02`.
  * `robust` (optimisation): 1st-order Roe scheme without MUSCL (piecewise-constant reconstruction) instead of
    JST; FFD `bspline_low`, 9 × 5 × 5 cubic B-spline control points instead of 15 × 5 × 5; nose clamped to 0th and
    1st order (planes i = 0, 1 carry no variables), tail to 0th order (i = 8); tighter second-difference limits;
    Sobolev metric `I + eps D'D` as the initial Hessian of the trust-region SQP (180 variables).
  * `verify` (final check): JST, 2nd order, recomputes the baseline and the final design of the robust run.
  * `v02`: the previous setup.
* `scripts/smoothness.py`: smoothness metrics from SU2 surface output — plane cuts of the surface triangulation,
  radius on five rays around the body axis, warts (non-monotone nose radius), waves of the wall displacement,
  pressure extrema; command line and `run.py` write `smooth.json`.
* `driver.Problem`: `numerics`, `clamp_i`, `sobolev` settings and `metric()`; `config/wing_body.cfg` takes the
  scheme from the preset (`{NUMERICS}`; default unchanged: JST). `trsqp.py` uses `Problem.metric()` when present.
* `tests/test_presets_smoothness.py` (no SU2 needed) and a GitHub Actions workflow running all unit tests.

## 0.2.0 (2026-09-28)

* `scripts/dakota_driver.py`: DAKOTA analysis driver (standard and aprepro parameters files, ASV values /
  gradients, DVV subsets) returning −K, 0.995 − V/V0 and CMy with analytic gradients from the two discrete
  adjoints and the volume integral; `setup` embeds the FFD box and writes the DAKOTA input (bounds,
  descriptors, linear smoothness constraints for the B-spline box); `--stub` analytic test problem.
  `dakota/trimmed_ld.in` (optpp_q_newton, 66 variables) and `dakota/README.md`. Checked with DAKOTA 6.24: the
  stub optimum equals SLSQP (optpp_q_newton, rol); two iterations with SU2 on the coarse wing-body mesh.
* `scripts/geometry_cadquery.py`: parametric fuselage (length, volume, exponent, base cut) and wing in CadQuery
  → STEP → Gmsh mesh with the markers and size fields of `make_mesh.py`; `cad/aircraft_baseline.step`,
  `figures/cadquery_mesh.png`.
* `make_mesh.py` split into `build_aircraft()` and `mesh_domain()` (same mesh, same command line);
  `driver.Evaluator.load_eval()` / `restore()` reload finished designs of a workdir (used by `trsqp.py` and the
  DAKOTA driver).
* `tests/`: unit tests for the DAKOTA driver and the CadQuery route, no SU2 needed.
* README: sections Optimizers, Geometry, Verification status, Tests.

## 0.1.1 (2026-09-28)

* Results of run 2 (15 × 5 × 5 B-spline FFD, 450 variables): L/D 15.91 → 21.49 (+35 %), trimmed, volume 99.5 %;
  stopped by the time budget with a 4.1 % KKT residual.
* `scripts/trsqp.py`: trust-region SQP with damped BFGS and a KKT-residual stopping test, used after SLSQP.
* `driver.py`: dCD/dCL and dCMy/dCL now come from a separate run at AoA + 0.1° instead of SU2's `EVAL_DOF_DCX`
  estimate, which can be stale after a restart (su2code/SU2#2937); the adjoint gets them with
  `DISCARD_INFILES= YES` (plus `EVAL_DOF_DCX= YES`, without which SU2 ignores the discard flag). Old behaviour:
  `--dcx su2`.
* `post.py`: optimization history per accepted iteration (SLSQP and TR-SQP), axis limits follow the shape.
* Figures redrawn from run 2 with English labels; README rewritten (requirements, how to run, results, citation);
  `requirements.txt` added; `CITATION.cff` lists SU2 and the HiSST 2018 paper.

## 0.1.0 (2026-09-28)

* First public version: SLSQP driver, discrete-adjoint gradients at fixed CL, CMy = 0 constraint, analytic
  volume gradient, axisymmetric benchmark, run 1 (66 variables, +15.4 %).
