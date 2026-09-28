# Changelog

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
