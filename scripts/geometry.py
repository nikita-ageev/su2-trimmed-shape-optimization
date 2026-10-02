"""Generic supersonic wing-body configuration (half model) and its FFD boxes.

Units: metres. x from nose to tail, y spanwise (y >= 0 is computed, y = 0 is the symmetry plane), z up.
"""
import numpy as np

# --- fuselage: Sears-Haack body (minimum linear-theory wave drag for given length and volume)
L_FUS = 60.0        # length
R_MAX = 1.6         # maximum radius (fineness ratio L/D = 18.75)

# --- wing: thin trapezoidal near-delta wing, biconvex section, NOT optimised
X_LE_ROOT = 24.0    # root leading edge (at the symmetry plane)
C_ROOT = 30.0       # root chord
SEMI_SPAN = 14.0    # half span
SWEEP_LE_DEG = 62.0 # leading-edge sweep (at M = 1.7 the Mach angle is 36 deg -> subsonic leading edge)
C_TIP = 3.0         # tip chord
T_C = 0.03          # thickness-to-chord ratio
Z_WING = 0.0        # mid wing

X_LE_TIP = X_LE_ROOT + SEMI_SPAN * np.tan(np.radians(SWEEP_LE_DEG))
LAMBDA = C_TIP / C_ROOT
S_HALF = 0.5 * (C_ROOT + C_TIP) * SEMI_SPAN                                  # half-wing reference area
MAC = 2.0 / 3.0 * C_ROOT * (1 + LAMBDA + LAMBDA**2) / (1 + LAMBDA)           # mean aerodynamic chord
Y_MAC = SEMI_SPAN / 3.0 * (1 + 2 * LAMBDA) / (1 + LAMBDA)
X_LE_MAC = X_LE_ROOT + Y_MAC * np.tan(np.radians(SWEEP_LE_DEG))

# centre of gravity: neutral point 43.16 m (from dCMy/dCL of the baseline) minus a 4 % MAC static margin
X_CG = 42.35


def sears_haack_r(x):
    xi = np.clip(np.asarray(x) / L_FUS, 0.0, 1.0)
    return R_MAX * (4.0 * xi * (1.0 - xi)) ** 0.75


V_FUS_FULL = 3.0 * np.pi**2 / 16.0 * R_MAX**2 * L_FUS   # analytic Sears-Haack volume
V_FUS_HALF = 0.5 * V_FUS_FULL

# --- FFD boxes around the fuselage. The outer layer j = J is frozen, so the wing panel outside the box does
# not move; the wing root inside the box deforms together with the fuselage (wing-body junction).
FFD_PRESETS = {
    # 11 x 3 x 2 Bernstein control points, 66 design variables (the published result)
    "bezier": dict(x=(-0.4, L_FUS + 0.4), y=(0.0, 2.6), z=(-2.0, 2.0), deg=(10, 2, 1), blend=None, fixj=2, def_nl=2),
    # 15 x 5 x 5 cubic uniform B-spline control points, richer cross-sections (use with smoothness constraints)
    "bspline": dict(x=(-0.4, L_FUS + 0.4), y=(0.0, 2.8), z=(-2.0, 2.0), deg=(14, 4, 4), blend=(4, 4, 4), fixj=4, def_nl=3),
    # v0.3 "robust" preset: 9 x 5 x 5 cubic B-spline control points (half as many planes along x, still C2);
    # the nose planes i = 0, 1 and the tail plane i = 8 are clamped by the preset (scripts/presets.py)
    "bspline_low": dict(x=(-0.4, L_FUS + 0.4), y=(0.0, 2.8), z=(-2.0, 2.0), deg=(8, 4, 4), blend=(4, 4, 4), fixj=4, def_nl=3),
}

if __name__ == "__main__":
    print(f"X_LE_TIP={X_LE_TIP:.2f}  S_half={S_HALF:.1f} m2  MAC={MAC:.2f} m  X_LE_MAC={X_LE_MAC:.2f}  Y_MAC={Y_MAC:.2f}")
    print(f"V_fus_half={V_FUS_HALF:.2f} m3  25% MAC at x={X_LE_MAC + 0.25 * MAC:.2f}")
