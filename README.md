# Trimmed shape optimization with the SU2 discrete adjoint

A Python driver around [SU2](https://su2code.github.io) 8.5.0 that maximizes the lift-to-drag ratio of an
aircraft shape **at fixed lift and zero pitching moment** (trimmed flight):

* **fixed lift:** every flow solution uses SU2's fixed-CL mode, and the gradients are taken at constant CL;
* **trim:** the pitching moment about the centre of gravity is an equality constraint, `CMy = 0`, with its own
  discrete-adjoint gradient;
* **volume:** fuselage volume `V >= 0.995 V0`; the driver computes the volume and its analytic gradient from the
  surface mesh;
* **length:** fixed (FFD control points move only in y and z).

Shape parameterization: FFD (free-form deformation). Gradients: SU2 discrete adjoint (algorithmic
differentiation), checked against finite differences. Optimizers: SciPy SLSQP and a small trust-region SQP
(`scripts/trsqp.py`).

**Main result** (supersonic wing-body, M = 1.7, Euler, CL = 0.10, 450 design variables):
L/D goes from **15.91** (baseline, not trimmed) to **21.49** trimmed (**+35 %**), `CMy = -3·10⁻⁷`,
fuselage volume 99.50 %. On a 2.8× finer mesh the gain is +30.9 %. The run was stopped by its time budget; the
final KKT (first-order optimality) residual is 4.1 %, so the result is an improved design, **not a converged
optimum**.

## Contents

- [Problem](#problem) · [Results](#results) · [Requirements](#requirements) · [Installation](#installation)
- [Quick start: axisymmetric benchmark](#quick-start-axisymmetric-benchmark-about-10-minutes) ·
  [Wing-body case](#wing-body-case) · [Repository layout](#repository-layout)
- [Practical notes](#practical-notes) · [Limitations](#limitations) · [Citing](#citing)

## Problem

Design variables `p` are FFD control-point displacements. With the target lift coefficient `CL*` and the centre
of gravity `x_cg`:

```
maximize    K(p) = CL / CD
subject to  CL(p, alpha) = CL*          (alpha is found by SU2 in fixed-CL mode)
            CMy(p; x_cg)  = 0           (trim, equality constraint)
            V(p)         >= 0.995 V0    (fuselage volume)
            L(p)          = L0          (by construction: no x-displacements)
            |p_i| <= b_i                (deformation bounds)
            |second differences of the displacements| <= s   (smoothness, B-spline box only; linear)
```

Because CL is fixed, maximizing K is the same as minimizing CD, so the driver minimizes CD.

**Gradients at constant lift.** With the angle of attack as a dependent variable, the total derivative is

```
dJ/dp |_(CL = CL*) = dJ/dp - (dJ/dCL) * dCL/dp,     J = CD or CMy
```

The SU2 discrete adjoint (`SU2_CFD_AD`) computes exactly this when it runs in fixed-CL mode: it differentiates
`J - (dJ/dCL) CL`, with `dJ/dCL` taken from `DCD_DCL_VALUE` / `DCMY_DCL_VALUE`. The driver computes these two
numbers with one extra flow solution at `alpha + 0.1°` (restarted from the converged solution) and passes them to
the adjoint with `DISCARD_INFILES= YES`. SU2's built-in estimate (`EVAL_DOF_DCX= YES`) is available with
`--dcx su2`, but after a restart it is sometimes not written to `flow.meta`, and a stale value then goes into
the gradient (seen here as `dCMy/dCL` jumping from −0.04 to +0.85; reported as
[su2code/SU2#2937](https://github.com/su2code/SU2/issues/2937)).

One gradient therefore costs one direct run, one run at `alpha + 0.1°`, two adjoint runs (CD and CMy) and two
`SU2_DOT_AD` projections onto the FFD variables. The cost does not depend on the number of design variables.

**Volume.** The half-model volume is a surface integral over the body, `V = ∮ y n_y dA` (the symmetry plane
y = 0 contributes nothing). The driver evaluates it on the deformed surface mesh with its own Python
implementation of the SU2 FFD map (Bernstein or uniform B-spline blending; matches `SU2_DEF` to about 1e-10 m).
The volume gradient is analytic: node derivatives of the integral, then the chain rule through the FFD basis.
Gmsh does not orient surface triangles consistently, so the driver orients each one by its owner tetrahedron.

**Trust-region SQP.** With 450 variables, SLSQP's line search stalled and its first step after a restart was
too large. `trsqp.py` solves a quadratic model with the linearized constraints and an ∞-norm trust region, accepts
steps on the merit function `1e4 CD + 20 |1e3 CMy| + 20 max(0, -c_V)`, and updates a damped BFGS Hessian of the
Lagrangian. The Lagrange multipliers come from sign-constrained least squares on the KKT conditions. Their
residual is reported at every step and is used as a stopping criterion.

## Results

**Wing-body, M = 1.7, Euler, fixed CL = 0.10.** Sears-Haack fuselage L = 60 m, D = 3.2 m; thin near-delta wing
with 62° leading-edge sweep (not optimized); half model, 347k tetrahedra; centre of gravity at 42.35 m, 4 % MAC
(mean aerodynamic chord) ahead of the neutral point, so the baseline is statically stable and not trimmed.

| | baseline | run 1: 66 variables | **run 2: 450 variables** |
|---|---|---|---|
| FFD box | | 11 × 3 × 2 Bernstein | 15 × 5 × 5 cubic B-spline, smoothness constraints |
| K = CL/CD | 15.91 (not trimmed) | 18.37 (+15.4 %) | **21.49 (+35.0 %)** |
| CD, counts (1e-4) | 62.8 | 54.5 | **46.5 (−25.9 %)** |
| CMy about the CG | −3.6·10⁻³ | −1·10⁻⁶ | **−3·10⁻⁷** |
| fuselage volume | 100 % | 99.50 % | 99.50 % (active) |
| angle of attack | 2.66° | 2.13° | 1.50° |
| finer mesh (0.8M / 0.96M tetrahedra) | | +14.1 % | +30.9 % (K 14.92 → 19.53) |
| optimizer | | 13 SLSQP iterations | 10 SLSQP + 18 trust-region SQP iterations |
| KKT residual at the end | | | 4.1 % (75 % at the start of TR-SQP) |

Trim comes from the fuselage itself, without a control surface: the camber of its mid line produces the nose-up
moment that balances the stability margin. The optimizer also removes the cross-section-area bump over the wing
(area rule) and moves volume forward.

Run 2 took 50 flow solutions (6.6 hours on 10 cores of an Apple M5 laptop) and was stopped by its time budget.
At the final design, 252 smoothness constraints, one bound and the volume constraint are active. The KKT residual
of 4.1 % means the design is close to, but not at, a stationary point.

**Gradient checks** (central differences, every point a fixed-CL solution):

* baseline, step 0.1 m: adjoint `dCD/dp` agrees with finite differences to 0.02 % and 0.009 % on two variables,
  `dCMy/dp` to 0.001 %; the analytic volume gradient agrees to 1e-13;
* final design of run 2, step 0.05 m: `dCMy/dp` 0.05 % and 0.67 %, `dCD/dp` 0.51 % and 3.8 %.

In run 2, the SLSQP phase (designs 0–29) still used SU2's `EVAL_DOF_DCX` estimate of `dJ/dCL` (see above). The
trust-region phase recomputed the gradient at its start point and used the separate `alpha + 0.1°` run from then
on. A rerun with the current code will therefore follow a different path.

![Optimization history of run 2: K, CMy and volume per accepted iteration](figures/opt_history.png)

![Side view in the symmetry plane before and after run 2](figures/shape_side.png)

*Side view in the symmetry plane: baseline (blue), optimized (red), mid line (dashed). Bottom: mid-line shift and
section-height change.*

![Cross-sections before and after run 2](figures/shape_sections.png)

![Cross-section area distribution before and after run 2](figures/area_distribution.png)

**Axisymmetric Sears-Haack body, M = 1.8, Euler** (L = 1, V = 0.01 L³, 24.8k triangles, 25 FFD variables,
volume ≥ baseline): CD **−1.29 %** at V/V0 = 1.00000 after 30 SLSQP iterations (82 solver calls, 10 minutes on
6 cores). The adjoint gradient agrees with central finite differences to a relative error of 1.5·10⁻⁵. Volume
moves from the nose to the aft body. In linear theory the Sears-Haack body is already optimal, so the Euler
optimizer can only exploit the nonlinear effects, and the gain is small.

![Axisymmetric benchmark: shape, radius change and Cp](figures/bench_axi_shape.png)

## Requirements

| component | version used | notes |
|---|---|---|
| SU2 | 8.5.0 | built with `-Denable-autodiff=true` (`SU2_CFD_AD`, `SU2_DOT_AD`) and MPI |
| Python | 3.12 | NumPy 2.5, SciPy 1.18 (SLSQP, `lsq_linear`), Matplotlib 3.11 |
| Gmsh | 4.15 with the Python API (conda-forge `gmsh`, `python-gmsh`) | mesh generation (OpenCASCADE kernel) |
| MPI | Open MPI 5.0 | optional; `SU2_NP=1` runs serially |
| VTK | optional | only for the Mach-number plot in `post.py` |

Tested on macOS (Apple silicon, arm64). The scripts contain nothing platform-specific; Linux should work with the
same environment.

## Installation

conda-forge has no SU2 package for macOS arm64, so SU2 is built from source (about 20 minutes):

```bash
mamba env create -f environment.yml          # Python, gmsh, SciPy + compilers, meson, ninja, Open MPI
conda activate su2

git clone --depth 1 --branch v8.5.0 https://github.com/su2code/SU2.git
cd SU2
export CC=mpicc CXX=mpicxx
./meson.py build -Denable-autodiff=true -Dwith-mpi=enabled -Denable-pywrapper=false \
    --prefix=$HOME/su2 --buildtype=release
./ninja -C build install

export SU2_RUN=$HOME/su2/bin                 # the driver looks for the binaries here, then in PATH
```

If SU2 with AD is already installed, the Python side alone is `pip install -r requirements.txt`.

Environment variables used by the scripts: `SU2_RUN` (SU2 bin directory), `SU2_NP` (MPI ranks, default 4;
`1` runs without MPI), `MPIRUN` (launcher, default `mpirun`).

## Quick start: axisymmetric benchmark (about 10 minutes)

From the repository root:

```bash
SU2_NP=4 python scripts/bench_axi.py --workdir runs/axi --maxiter 30
python scripts/bench_post.py --workdir runs/axi --out runs/axi/figures
```

The script generates the mesh with Gmsh, embeds the FFD box (`SU2_DEF`), solves the baseline and its adjoint,
checks the gradient by finite differences on two variables and runs SLSQP. Smoke test, about 4 minutes on
2 cores: `SU2_NP=2 python scripts/bench_axi.py --workdir runs/axi_smoke --maxiter 2` (from a clean clone: CD
−0.61 % after 2 iterations; adjoint and finite differences agree to 1.4·10⁻⁵).

Output in `runs/axi/`: `history_opt.csv` (CD, V/V0 per solver call), `result.json` (CD before/after, gradient
check, iterations, time), `x_opt.txt` (optimal FFD displacements), and one folder `e_NNN/` per design with the SU2
configs, logs, surface CSV and ParaView files.

## Wing-body case

Run 2 (B-spline box, 450 variables): SLSQP first, then the trust-region SQP from the last SLSQP iterate.

```bash
python scripts/make_mesh.py mesh/wing_body.su2 0.25      # ~350k tetrahedra, ~1 minute (0.6 -> ~90k, quick test)
SU2_NP=10 python scripts/driver.py --mesh mesh/wing_body.su2 --workdir runs/wb2 --ffd bspline --maxiter 10
SU2_NP=10 python scripts/trsqp.py --workdir runs/wb2 --maxiter 60 --budget-hours 6
SU2_NP=10 python scripts/fd_check.py --workdir runs/wb2 --eval <N> --h 0.05   # adjoint vs finite differences
python scripts/kkt.py --workdir runs/wb2                                       # KKT residual per iteration
python scripts/post.py --workdir runs/wb2 --out runs/wb2/figures               # PNG figures
```

Run 1 (Bernstein box, 66 variables): `driver.py --ffd bezier --maxiter 15` and no `trsqp.py`.

Options of `driver.py`: `--target-cl` (0.10), `--xcg` (42.35 m), `--vol-frac` (0.995), `--ffd bezier|bspline`,
`--bnd-z`, `--bnd-y`, `--maxiter`, `--budget-hours`, `--stop-dk` (stop when K changes by less than this fraction
on three accepted iterations with all constraints satisfied), `--dcx alpha|su2` (how `dJ/dCL` is computed),
`--template` (another SU2 config template). Options of `trsqp.py`: `--start-eval` (default: the final design of
`driver.py`), `--maxiter`, `--delta0` (initial trust radius, 0.15 m), `--nu` (merit penalty, 20), `--budget-hours`,
`--stop-dk`, `--kkt-tol` (0.01).

* `--ffd bezier`: 11 × 3 × 2 Bernstein control points, 66 variables (z-displacements of the layers j = 0, 1 and
  y-displacements of j = 1), bounds |Δz| ≤ 1 m, |Δy| ≤ 0.5 m.
* `--ffd bspline`: 15 × 5 × 5 cubic B-spline control points; the z-displacements of j = 0 and j = 1 are linked
  (smooth crest and keel in the symmetry plane); 450 variables mapped to 525 SU2 design variables; bounds
  |Δz| ≤ 1.5 m, |Δy| ≤ 0.8 m; 1110 linear smoothness constraints on second differences of the displacement field
  (0.40 m along x, 0.25 m along y and z).

In both cases the outer FFD layer is frozen: the wing panel outside the box does not move, and the wing root
inside the box deforms with the fuselage.

Output in the workdir: `dsn_history.csv` (CD, CL, CMy, AoA, K, V/V0 per design), `opt_iterations.txt` (accepted
iterations of both optimizers), `opt_result.json` and `trsqp_result.json` (summaries), `trsqp_log.csv` (every
trust-region step), `timing.csv` (every SU2 call), `kkt.json`, and one folder per design `dsn_NNN/` with `p.txt`,
`x.txt`, the SU2 configs and logs, `history*.csv`, `flow.meta`, `dcx.json` (dCD/dCL, dCMy/dCL), the surface CSV,
`flow.vtu` and the gradients `grad_CD.txt`, `grad_CMy.txt` (optimizer variables) and `gradx_*.txt` (SU2
variables).

## Repository layout

| path | content |
|---|---|
| `scripts/driver.py` | problem set-up, evaluator with caching and the dJ/dCL run, SLSQP, stopping rule |
| `scripts/trsqp.py` | trust-region SQP on the same problem and evaluator |
| `scripts/su2run.py` | SU2 wrapper: config templates, runs, history and gradient readers, FFD re-implementation, volume and its gradient |
| `scripts/geometry.py`, `scripts/make_mesh.py` | wing-body geometry, FFD boxes, Gmsh mesh (markers `aircraft`, `symmetry`, `farfield`) |
| `scripts/fd_check.py`, `scripts/kkt.py`, `scripts/post.py` | gradient check, KKT check, figures |
| `scripts/bench_axi*.py` | axisymmetric Sears-Haack benchmark: mesh, optimization, figures |
| `config/*.cfg` | SU2 config templates with `{PLACEHOLDERS}` |
| `figures/` | figures of the published runs |

Meshes, solutions and restart files are generated by the scripts and are not stored in the repository.

## Practical notes

* Paths inside SU2 config files must not contain spaces, so every run folder uses relative paths and the input
  mesh is copied into `<workdir>/setup/`.
* After `SU2_CFD_AD`, `restart_adj_<obj>.dat` has to be copied to `solution_adj_<obj>.dat` for `SU2_DOT_AD`.
* SLSQP modifies its argument in place; the design cache stores copies.
* Each new design restarts from the previous flow solution and angle of attack.
* When the adjoint restarts from a fixed-CL solution, SU2 reads `DCD_DCL_VALUE` / `DCMY_DCL_VALUE` from the restart
  metadata unless `DISCARD_INFILES= YES` is set, and for a fixed-CL adjoint SU2 resets `DISCARD_INFILES` to `NO`
  unless `EVAL_DOF_DCX= YES` is also set. The driver sets both in the adjoint configs; the log then shows
  "Discarding the dCD/dCL in the direct solution file".

## Limitations

* Inviscid (Euler) flow: no friction and no base drag; absolute K values are inviscid. A RANS run with the discrete
  adjoint is the natural next step (SU2 differentiates the turbulence model as well).
* Only the fuselage is shaped; the wing planform, twist and camber are fixed.
* Run 2 is not converged to a KKT point (4.1 % residual); run 1 ended with 45 of 66 variables on their bounds, so
  its gain was limited by the allowed deformation.
* Cabin-size constraints per station are not included (only the total volume).
* The optimization mesh is not grid-converged in absolute CD (about 6 % between 0.35M and 0.8M cells); the gain
  and the trim carry over to the finer meshes.
* The FFD re-implementation assumes a rectangular, axis-aligned box.

## Related

* SU2: <https://github.com/su2code/SU2>, documentation <https://su2code.github.io>.
* Upstream fixes and reports found while building this:
  [su2code/SU2#2934](https://github.com/su2code/SU2/pull/2934) (SU2_PY station names for `SU2_GEO` constraints),
  [su2code/SU2#2935](https://github.com/su2code/SU2/pull/2935) (quoting the SU2 executable path in SU2_PY),
  [su2code/SU2#2937](https://github.com/su2code/SU2/issues/2937) (dCL derivatives not written to `flow.meta`).
* Earlier work on the same Sears-Haack drag problem at M = 1.8: a combined approach (global IOSO search followed
  by continuous-adjoint SU2 refinement), N. D. Ageev, A. A. Pavlenko, HiSST 2018 (below). The axisymmetric
  benchmark here repeats that problem with the discrete adjoint (Euler only, fixed ends).

## Citing

If this driver is useful, please cite the repository (see `CITATION.cff`) and SU2:

* N. Ageev, *su2-trimmed-shape-optimization: trimmed aerodynamic shape optimization with the SU2 discrete
  adjoint*, version 0.1.1, 2026, <https://github.com/nikita-ageev/su2-trimmed-shape-optimization>.
* T. D. Economon, F. Palacios, S. R. Copeland, T. W. Lukaczyk, J. J. Alonso, *SU2: An Open-Source Suite for
  Multiphysics Simulation and Design*, AIAA Journal 54(3), 828–846, 2016,
  [doi:10.2514/1.J053813](https://doi.org/10.2514/1.J053813).

Related earlier paper by the author:

* N. D. Ageev, A. A. Pavlenko, *Combined approach on the aerodynamic shape optimization*, HiSST 2018:
  International Conference on High-Speed Vehicle Science & Technology, Moscow, 26–29 November 2018,
  [PDF](https://aerospacerepository.org/wp-content/uploads/2023/10/hisst-2018_600969.pdf).

## License

MIT (see `LICENSE`). SU2 itself is not included; it is distributed under LGPL-2.1.

---

## Кратко по-русски

Драйвер оптимизации формы для SU2 8.5.0 с дискретным сопряжённым методом: максимум аэродинамического качества
K = CL/CD при заданной подъёмной силе (режим фиксированного CL) и нулевом моменте тангажа относительно центра масс
(балансировка), объём фюзеляжа не меньше 99,5 %, длина фиксирована. Компоновка «крыло–фюзеляж», M = 1,7, уравнения
Эйлера, 450 параметров FFD (свободной деформации): K 15,91 → 21,49 (+35 %), CMy = −3·10⁻⁷, объём 99,50 %; на мелкой
сетке +30,9 %. Остановлено по бюджету времени, невязка условий оптимальности (ККТ) 4,1 % — улучшенная форма,
а не строгий оптимум.
