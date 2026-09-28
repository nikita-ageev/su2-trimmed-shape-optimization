# Changelog

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
