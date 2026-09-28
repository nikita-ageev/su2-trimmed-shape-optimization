# Running the optimization with DAKOTA

[DAKOTA](https://dakota.sandia.gov) (Sandia, open source, LGPL) can drive the same trimmed L/D problem as
`scripts/driver.py` through the analysis driver `scripts/dakota_driver.py`: DAKOTA proposes the FFD
control-point displacements, the driver runs SU2 (fixed-CL flow solution, two discrete adjoints) and returns
the objective, the two constraints and their analytic gradients.

```
DAKOTA  --(params file: p, ASV, DVV)-->  dakota_driver.py  -->  SU2_DEF, SU2_CFD, SU2_CFD_AD x2, SU2_DOT_AD x2
        <--(results file: f, g, h, gradients)--
```

| response | meaning | DAKOTA type |
|---|---|---|
| `minus_K = -CL/CD` | negative lift-to-drag ratio at the target CL | objective function (minimized) |
| `vol_deficit = 0.995 - V/V0` | fuselage volume deficit | nonlinear inequality, `<= 0` (DAKOTA default bounds) |
| `CMy` | pitching moment about the centre of gravity | nonlinear equality, target 0 |

Gradients: `d(-K)/dp = (CL/CD^2) dCD/dp` with `dCD/dp` from the fixed-CL adjoint, `dCMy/dp` from the second
adjoint, `d(V/V0)/dp` analytic (`su2run.Volume`). The design variables, bounds and the linear smoothness
constraints of the B-spline box are exactly those of `driver.py` (`Problem.pvars`, `Problem.bounds()`,
`Problem.Dsm`). Hessians are not available: DAKOTA must use a quasi-Newton or SQP method.

## How to run

From the repository root (relative paths in the input file are relative to the DAKOTA working directory):

```bash
python scripts/make_mesh.py mesh/wing_body.su2 0.25            # or the CadQuery route, see the main README
python scripts/dakota_driver.py setup --mesh mesh/wing_body.su2 --workdir runs/dk --ffd bezier \
    --out dakota/trimmed_ld.in                                 # embeds the FFD box, writes the input file
SU2_NP=8 dakota -i dakota/trimmed_ld.in -o runs/dk/dakota.out  # runs the optimization
```

`setup` writes `<workdir>/setup/` (FFD-embedded mesh, `settings.json`) in the same format as `driver.py`, so
`fd_check.py`, `kkt.py` and `post.py` work on a DAKOTA workdir, and a workdir started by `driver.py` can be
continued with DAKOTA. `dakota/trimmed_ld.in` in this folder is the generated file for the Bernstein box
(66 variables; the published run 1). For the B-spline box use `--ffd bspline`: 450 variables and 1110 linear
smoothness rows go into the input file (about 5 MB), which DAKOTA handles as `linear_inequality_constraint_matrix`.

Options of `setup`: `--method optpp_q_newton|rol`, `--maxiter`, `--target-cl`, `--xcg`, `--vol-frac`,
`--python` (interpreter name in the `analysis_drivers` string, default `python`).

Every SU2 design goes to `<workdir>/dsn_NNN/` as with `driver.py`; `dakota_evals.csv` maps DAKOTA evaluation
ids to design folders, `dsn_history.csv` has CD, CL, CMy, K and V/V0 per design, and `dakota_tabular.dat` is
DAKOTA's own table. The driver is a new process for every evaluation, reloads the finished designs of the
workdir and serves repeated requests (value first, gradient later) from that cache; SU2 restarts from the
previous solution. Evaluations must run one at a time (no `asynchronous` in the interface block).

Environment: `SU2_RUN`, `SU2_NP`, `MPIRUN` as for `driver.py`; the MPI launcher must be on `PATH`. OPT++
writes its own log `OPT_DEFAULT.out` into the working directory (ignored by git). The repository path itself
may contain blanks; the paths inside the input file (workdir, driver command) must not, because DAKOTA's
`fork` interface splits the driver string on blanks.

## Methods

The public DAKOTA build (6.24, `dakota-6.24.0-public-*-cli`) ships OPT++, ROL and CONMIN; NPSOL, NLPQL and
DOT are commercial libraries and are not included, so `npsol_sqp`, `nlpql_sqp` and `dot_sqp` are not
available without a custom build. With `analytic_gradients` and the general constraints of this problem:

| method | constraints | status |
|---|---|---|
| `optpp_q_newton` (quasi-Newton, nonlinear interior point for the constraints) | bounds, linear, nonlinear equality and inequality | default in `trimmed_ld.in`; run on the stub (optimum = SLSQP) and with SU2 on the coarse mesh |
| `rol` (Trilinos ROL, augmented Lagrangian) | same | run on the stub (`setup --method rol`): optimum = SLSQP within 3e-7, but 1524 evaluations against about 30 for optpp |
| `conmin_mfd` (method of feasible directions) | DAKOTA passes the equality constraint to CONMIN | runs on the stub, stops after 9 evaluations 9 % above the optimum; not recommended |

Response scaling (`scaling` in the method block, `primary_scales` 20, inequality 0.01, equality 1e-3) brings
the three responses to order one, as `F_SCALE` / `C_SCALE` do in `driver.py`.

## Testing without SU2

`--stub` replaces SU2 by an analytic problem with the same interface (convex CD, linear CMy and volume):

```bash
python scripts/dakota_driver.py setup --stub --stub-n 6 --workdir runs/stub --out runs/stub/stub.in
dakota -i runs/stub/stub.in
python -m unittest discover -s tests -v          # file formats, ASV/DVV handling, gradient assembly
```

The unit tests in `tests/test_dakota_driver.py` cover both parameters-file formats (standard and aprepro),
the ASV combinations (values only, gradients only, mixed), DVV subsets, the rejection of Hessian requests,
and compare the gradients written to the results file with finite differences of the stub. If a `dakota`
executable is on `PATH` (or in `$DAKOTA_EXE`), the last test runs DAKOTA on the stub and checks the optimum
against SciPy SLSQP.

## What has been verified

* File formats and gradient assembly: unit tests (no SU2).
* DAKOTA 6.24 (macOS arm64 binary) with `optpp_q_newton` on the stub: converges to the SLSQP optimum
  (|CMy| < 1e-5, volume constraint satisfied, variables within 2e-3).
* DAKOTA 6.24 with `optpp_q_newton` and SU2 8.5.0 on the coarse wing-body mesh (`make_mesh.py ... 0.6`,
  93k tetrahedra, 66 variables, `SU2_NP=8`): two iterations, three SU2 designs with gradients in 100 s
  (about 55 s per design: direct run, run at AoA + 0.1°, two adjoints, two projections). DAKOTA's first
  request (p = 0, value and gradient) was served from a design computed before by a manual driver call, which
  exercises the cache; the next two designs moved CMy from −4.56e-3 to −3.77e-3 and V/V0 from 1.0000 to 1.0003
  at K 19.61 → 19.58 (the interior-point method restores feasibility first). Numbers in the main README.
* Not verified: a full DAKOTA optimization on the published mesh (0.25 m, 350k tetrahedra) and the B-spline
  case with the 1110 linear constraints (input generated, not run).
