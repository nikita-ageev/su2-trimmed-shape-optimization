"""Run presets (v0.3): what the optimisation run uses and what the final check uses.

robust - the default for optimisation
    numerics  1st-order upwind (Roe, MUSCL_FLOW= NO): piecewise-constant (0th-order) reconstruction of the state
              at cell faces -> a 1st-order scheme. Monotone at shocks, no limiter switching, so the discrete
              adjoint gradient varies smoothly with the shape, and the solver converges in fewer iterations.
    shape     FFD "bspline_low": 9 x 5 x 5 cubic B-spline control points (15 x 5 x 5 in v0.2), still C2 along x;
              nose clamped to 0th and 1st order (control planes i = 0 and i = 1 carry no variables: the tip
              position and the tip slope stay as built), tail clamped to 0th order (i = 8);
              tighter second-difference limits of the control net;
              Sobolev metric M = I + eps_x Dx'Dx + eps_yz Dyz'Dyz as the initial Hessian of trsqp.py
              (D = second differences of the control net).
verify - the final check only
    numerics  JST (2nd order, as in v0.2) on the same mesh: the baseline and the final design of the robust run are
              recomputed at the same CL; no optimisation.
v02     - the v0.2 setup (JST, 15 x 5 x 5, no clamps, no metric), for reproducing the published run.

Why: with 15 free control planes, free nose planes and an identity initial Hessian the optimiser moved neighbouring
control points in opposite directions (zig-zag of the control net -> waves of the surface with a period of two
control planes) and pushed the nose planes, which carry the largest pressure gradients, to their bounds (bumps on
the nose). See README, section "Robust preset".
"""
import numpy as np

NUMERICS = {
    "first_order": ("CONV_NUM_METHOD_FLOW= ROE\n"
                    "MUSCL_FLOW= NO\n"
                    "ENTROPY_FIX_COEFF= 0.05"),
    "second_order": ("CONV_NUM_METHOD_FLOW= JST\n"
                     "JST_SENSOR_COEFF= ( 0.5, 0.02 )"),
}

PRESETS = {
    "robust": dict(ffd="bspline_low", numerics="first_order", clamp_i=(0, 1, 8), sobolev=(2.0, 0.5),
                   direct_minval=-9.0, adj_minval=-7.5),
    "verify": dict(numerics="second_order", direct_minval=-10.5),
    "v02": dict(ffd="bspline", numerics="second_order", clamp_i=(), sobolev=None),
}


def problem_kwargs(name):
    """Keyword arguments of driver.Problem for a preset (verify takes the design space of the robust run)."""
    p = dict(PRESETS[name])
    p["preset"] = name
    return p


def second_differences(pvars, axis):
    """Rows of the second-difference operator of every variable family (direction, other two indices) along one
    lattice axis (0 = i along x, 1 = j, 2 = k). pvars: list of (dirn, i, j, k)."""
    idx = {}
    for n, v in enumerate(pvars):
        idx.setdefault(v[0], {})[tuple(v[1:4])] = n
    rows = []
    for fam in idx.values():
        for ijk, n in fam.items():
            m, p = list(ijk), list(ijk)
            m[axis] -= 1
            p[axis] += 1
            if tuple(m) in fam and tuple(p) in fam:
                r = np.zeros(len(pvars))
                r[fam[tuple(m)]] += 1.0
                r[n] -= 2.0
                r[fam[tuple(p)]] += 1.0
                rows.append(r)
    return np.array(rows).reshape(-1, len(pvars))


def sobolev_metric(pvars, sobolev):
    """M = I + eps_x Dx'Dx + eps_yz (Dy'Dy + Dz'Dz); identity when sobolev is None."""
    n = len(pvars)
    M = np.eye(n)
    if not sobolev:
        return M
    eps_x, eps_yz = sobolev
    Dx = second_differences(pvars, 0)
    M += eps_x * Dx.T @ Dx
    for ax in (1, 2):
        D = second_differences(pvars, ax)
        M += eps_yz * D.T @ D
    return M
