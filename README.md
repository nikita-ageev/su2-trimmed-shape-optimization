# Trimmed shape optimisation with the SU2 discrete adjoint

A small Python driver around [SU2](https://su2code.github.io) 8.5 that maximises the lift-to-drag ratio of an
aircraft shape **while keeping the aircraft trimmed**:

* **lift is preserved**: every flow solution is computed in SU2's fixed-CL mode (`FIXED_CL_MODE= YES`), the
  angle of attack is adjusted to the target CL, and the gradients are taken at constant CL;
* **the pitching moment about the centre of gravity is zero**: `CMy = 0` is an equality constraint with its own
  discrete-adjoint gradient;
* **volume is kept**: `V >= 0.995 V0`, volume and its gradient are computed by the driver from the surface mesh;
* **length is fixed**: the FFD control points move only in y and z.

Shape parameterisation is FFD (free-form deformation), the optimiser is SLSQP from SciPy, the gradients of CD and
CMy come from the SU2 discrete adjoint (algorithmic differentiation), and they are checked against finite
differences. Two examples are included: a quick axisymmetric Sears-Haack body at M = 1.8 (about 10 minutes on a
laptop) and a supersonic wing-body configuration at M = 1.7.

## Problem statement

Design variables `p` are FFD control-point displacements. With the target lift coefficient `CL*` and the centre
of gravity `x_cg`:

```
maximise    K(p) = CL / CD
subject to  CL(p, alpha) = CL*          (alpha is found by SU2 in fixed-CL mode)
            CMy(p; x_cg)  = 0           (trim, equality constraint)
            V(p)         >= 0.995 V0    (fuselage volume)
            L(p)          = L0          (by construction: no x-displacements)
            |p_i| <= b_i                (deformation bounds)
```

Because CL is held fixed, maximising K is the same as minimising CD, so the driver minimises CD.

**Gradients at constant lift.** In fixed-CL mode the direct run of SU2 ends with a finite difference in the angle
of attack and stores `dCD/dCL` and `dCMy/dCL` in `flow.meta`. The discrete adjoint (`SU2_CFD_AD`) reads them and
differentiates `J - (dJ/dCL) CL`, which gives the total derivative with the angle of attack as a dependent
variable:

```
dJ/dp |_(CL = CL*) = dJ/dp - (dJ/dalpha / dCL/dalpha) * dCL/dp,     J = CD or CMy
```

One design iteration therefore costs one direct run plus two adjoint runs (for CD and for CMy) plus two
`SU2_DOT_AD` projections onto the FFD variables, independent of the number of design variables.

**Volume.** The half-model volume is a surface integral over the body, `V = ∮ y n_y dA` (the symmetry plane
y = 0 contributes nothing). The driver evaluates it on the deformed surface mesh using its own Python
implementation of the SU2 FFD map (Bernstein or uniform B-spline blending; it matches `SU2_DEF` to about 1e-10 m),
so the volume gradient is analytic: node derivatives of the integral, then the chain rule through the FFD basis.
Gmsh does not orient surface triangles consistently, so they are oriented by their owner tetrahedron.

## Results

**Wing-body, M = 1.7, Euler, fixed CL = 0.10** (Sears-Haack fuselage L = 60 m, D = 3.2 m; thin 62° swept
near-delta wing that is not optimised; half model, 347k tetrahedra; 66 FFD variables around the fuselage;
centre of gravity at 42.35 m, 4 % MAC ahead of the neutral point):

| | baseline | optimised | change |
|---|---|---|---|
| K = CL/CD | 15.91 (not trimmed) | **18.37** | **+15.4 %** |
| CD, counts | 62.84 | 54.45 | −13.4 % |
| CL | 0.100 | 0.100 | held by fixed-CL mode |
| CMy about the CG | −3.6·10⁻³ | **−1·10⁻⁶** | trimmed without a control surface |
| fuselage volume | 100 % | 99.50 % | constraint active |
| length | 60.000 m | 60.000 m | fixed |
| angle of attack | 2.66° | 2.13° | |

13 SLSQP iterations, every first trial step accepted (14 direct and 13 pairs of adjoint solutions, about 5 minutes
per iteration on 10 cores). The trim comes from the fuselage itself: the camber of its mid line produces the
nose-up moment that balances the stability margin.

* **Gradient check** (central differences, step 0.1 m, baseline shape, every point a fixed-CL solution): the
  adjoint `dCD/dp` differs from finite differences by **0.02 %** and 0.009 % on the two tested variables,
  `dCMy/dp` by 0.001 %; the analytic volume gradient agrees to 1e-13.
* **Finer mesh** (0.8M tetrahedra, same FFD deformation): K 15.05 → 17.16, **+14.1 %**, CMy = −2.6·10⁻⁵.
  Absolute CD differs by about 6 % between the meshes, but the gain and the trim carry over.

![Optimisation history](figures/opt_history.png)

![Fuselage contour in the symmetry plane before and after (axis labels in Russian)](figures/shape_side.png)

*Side view in the symmetry plane: baseline (blue), optimised (red), mid line (dashed); bottom: mid-line shift and
section-height change. Axis labels are in Russian.*

**Axisymmetric Sears-Haack body, M = 1.8, Euler** (L = 1, V = 0.01 L³, 24.8k triangles, 25 FFD variables,
volume ≥ baseline): CD **−1.29 %** at V/V0 = 1.00000 in 30 SLSQP iterations (82 solver calls, 10 minutes on
6 cores); the adjoint gradient agrees with central finite differences to a relative error of 1.5·10⁻⁵. Volume
moves from the nose to the aft body. The Sears-Haack body is optimal in linear theory, so only the nonlinear
remainder is left for the Euler optimiser.

![Axisymmetric benchmark (labels in Russian)](figures/bench_axi_shape.png)

## Installation

SU2 must be built with algorithmic differentiation (`SU2_CFD_AD`, `SU2_DOT_AD`). On macOS arm64 there is no
conda-forge package, so build from source (about 20 minutes):

```bash
mamba env create -f environment.yml          # Python, gmsh, SciPy + compilers, meson, ninja, OpenMPI
conda activate su2

git clone --depth 1 --branch v8.5.0 https://github.com/su2code/SU2.git
cd SU2
export CC=mpicc CXX=mpicxx
./meson.py build -Denable-autodiff=true -Dwith-mpi=enabled -Denable-pywrapper=false \
    --prefix=$HOME/su2 --buildtype=release
./ninja -C build install

export SU2_RUN=$HOME/su2/bin                 # the driver looks for the binaries here, then in PATH
```

Environment variables used by the scripts: `SU2_RUN` (SU2 bin directory), `SU2_NP` (MPI ranks, default 4;
`1` runs without MPI), `MPIRUN` (launcher, default `mpirun`).

## Quick start: axisymmetric benchmark (about 10 minutes)

From the repository root:

```bash
SU2_NP=4 python scripts/bench_axi.py --workdir runs/axi --maxiter 30
python scripts/bench_post.py --workdir runs/axi --out runs/axi/figures
```

The script generates the mesh with gmsh, embeds the FFD box (`SU2_DEF`), solves the baseline and its adjoint,
checks the gradient by finite differences on two variables and runs SLSQP. A 2-iteration smoke test takes
about 4 minutes on 2 cores: `SU2_NP=2 python scripts/bench_axi.py --workdir runs/axi_smoke --maxiter 2`
(from a clean clone it gives CD −0.61 % after 2 iterations, adjoint vs finite differences 1.4·10⁻⁵).

Output in `runs/axi/`: `history_opt.csv` (CD, V/V0 per solver call), `result.json` (CD before/after, gradient
check, iterations, time), `x_opt.txt` (optimal FFD displacements), one folder `e_NNN/` per design with the SU2
configs, logs, surface CSV and ParaView files.

## Wing-body case

```bash
python scripts/make_mesh.py mesh/wing_body.su2 0.25      # ~350k tetrahedra, ~1 minute (0.6 -> ~90k, quick test)
SU2_NP=8 python scripts/driver.py --mesh mesh/wing_body.su2 --workdir runs/wing_body --maxiter 15
SU2_NP=8 python scripts/fd_check.py --workdir runs/wing_body --eval 0 --h 0.1    # adjoint vs finite differences
python scripts/kkt.py --workdir runs/wing_body                                    # KKT residual per iteration
python scripts/post.py --workdir runs/wing_body --out runs/wing_body/figures       # PNG figures
```

Options of `driver.py`: `--target-cl` (0.10), `--xcg` (42.35 m), `--vol-frac` (0.995), `--ffd bezier|bspline`,
`--bnd-z`, `--bnd-y`, `--maxiter`, `--budget-hours`, `--stop-dk` (stop when K changes by less than this fraction on
three accepted iterations with all constraints satisfied), `--template` (another SU2 config template).

* `--ffd bezier` (default, used for the results above): 11 × 3 × 2 Bernstein control points, 66 variables
  (z-displacements of the layers j = 0, 1 and y-displacements of j = 1), bounds |Δz| ≤ 1 m, |Δy| ≤ 0.5 m.
* `--ffd bspline` (experimental): 15 × 5 × 5 cubic B-spline control points, z-displacements of j = 0 and j = 1
  linked, 450 variables, bounds 1.5 / 0.8 m and linear smoothness constraints on second differences of the
  displacement field.

In both cases the outer FFD layer is frozen, so the wing panel outside the box does not move and the wing root
inside the box deforms with the fuselage.

Output in `runs/wing_body/`: `dsn_history.csv` (CD, CL, CMy, AoA, K, V/V0 per design), `opt_iterations.txt`
(accepted SLSQP iterations), `opt_result.json` (summary), `timing.csv` (every SU2 call), and one folder per design
`dsn_NNN/` with `p.txt`, `x.txt`, the SU2 configs and logs, `history*.csv`, `flow.meta`, the surface CSV,
`flow.vtu` and the gradients `grad_CD.txt`, `grad_CMy.txt` (optimiser variables) and `gradx_*.txt` (SU2 variables).

## Repository layout

| path | content |
|---|---|
| `scripts/driver.py` | the optimisation driver (problem set-up, evaluator with caching, SLSQP, stopping rule) |
| `scripts/su2run.py` | SU2 wrapper: config templates, runs, history/gradient readers, FFD re-implementation, volume and its gradient |
| `scripts/geometry.py`, `scripts/make_mesh.py` | wing-body geometry, FFD boxes, gmsh mesh (markers `aircraft`, `symmetry`, `farfield`) |
| `scripts/fd_check.py`, `scripts/kkt.py`, `scripts/post.py` | gradient check, KKT check, figures |
| `scripts/bench_axi*.py` | axisymmetric Sears-Haack benchmark: mesh, optimisation, figures |
| `config/*.cfg` | SU2 config templates with `{PLACEHOLDERS}` |
| `figures/` | figures of the published run |

Meshes, solutions and restart files are generated and are not stored in the repository.

## Practical notes

* Paths inside SU2 config files must not contain spaces, so every run folder uses relative paths and the input
  mesh is copied into `<workdir>/setup/`.
* After `SU2_CFD_AD`, the file `restart_adj_<obj>.dat` has to be copied to `solution_adj_<obj>.dat` for
  `SU2_DOT_AD`.
* SLSQP modifies its argument in place; the design cache stores copies.
* Each new design restarts from the previous flow solution and angle of attack.

## Limitations

* Inviscid (Euler) flow: no friction and no base drag; absolute K values are inviscid. A RANS run with the discrete
  adjoint is the natural next step (SU2 differentiates the turbulence model as well).
* Only the fuselage is shaped; the wing planform, twist and camber are fixed.
* With the 66-variable box, 45 variables end on their bounds: the gain is limited by the allowed deformation.
  Cabin-size constraints per station are not included (only the total volume).
* The coarse mesh is not grid-converged in absolute CD (about 6 % between 0.35M and 0.8M cells).
* The FFD re-implementation assumes a rectangular, axis-aligned box.

## Related

* SU2: <https://github.com/su2code/SU2>, documentation <https://su2code.github.io>.
* Two small upstream fixes found while building this:
  [su2code/SU2#2934](https://github.com/su2code/SU2/pull/2934) (SU2_PY station names for `SU2_GEO` constraints) and
  [su2code/SU2#2935](https://github.com/su2code/SU2/pull/2935) (quoting the SU2 executable path in SU2_PY).
* Background: an earlier combined approach (global IOSO search followed by a continuous-adjoint SU2 refinement)
  for the same Sears-Haack drag problem at M = 1.8:
  N. D. Ageev, A. A. Pavlenko, *Combined approach on the aerodynamic shape optimization*, HiSST 2018
  (International Conference on High-Speed Vehicle Science & Technology), Moscow, 26–29 November 2018,
  [PDF](https://aerospacerepository.org/wp-content/uploads/2023/10/hisst-2018_600969.pdf).
  The axisymmetric benchmark here repeats that problem with the discrete adjoint (Euler only, fixed ends).

## Citing

If this driver is useful, please cite the repository (see `CITATION.cff`) and SU2
(T. D. Economon et al., *SU2: An Open-Source Suite for Multiphysics Simulation and Design*, AIAA Journal 54(3),
2016).

## License

MIT (see `LICENSE`). SU2 itself is not included; it is distributed under LGPL-2.1.

---

## Кратко по-русски

Драйвер оптимизации формы для SU2 8.5 с дискретным сопряжённым методом: максимизация аэродинамического качества
K = CL/CD **при сохранении подъёмной силы** (режим фиксированного CL в SU2, градиенты берутся при постоянном CL)
и **балансировке по тангажу** (момент CMy относительно центра масс равен нулю — ограничение-равенство со своим
сопряжённым градиентом). Кроме того: объём фюзеляжа не меньше 99,5 % исходного (объём и его градиент считаются
по поверхностной сетке), длина фиксирована (контрольные точки FFD — свободной деформации — двигаются только по y
и z). Оптимизатор — SLSQP (последовательное квадратичное программирование) из SciPy.

Результат на компоновке «крыло–фюзеляж» при M = 1,7 (уравнения Эйлера, CL = 0,10): K 15,91 → 18,37 (+15,4 %),
CMy = −1·10⁻⁶, объём 99,50 %, длина 60,000 м. Градиент сопряжённого совпадает с конечными разностями до 0,02 %.
На мелкой сетке (0,8 млн ячеек) выигрыш +14,1 %. Быстрый пример — осесимметричное тело Сирса–Хаака при M = 1,8:
`SU2_NP=4 python scripts/bench_axi.py --workdir runs/axi`, около 10 минут. Подписи на двух рисунках — на русском.
