"""
Props 2 and 4 dead. Closed-form equilibrium (eq 22-25):
    f1_bar = f3_bar = m g / 2
    omega1_bar = omega3_bar = -sqrt(m g / (2 kf))
    omega_B_bar = (0, 0, ktau m g / gamma)
    n_bar = (0, 0, 1)

Single input u1 = (f3-f3bar) - (f1-f1bar). Total thrust constraint f1+f3 = fSigma_cmd.
LQR weights per paper: 0 on (p,q), 1000 on nx, 2 on ny, 0 on motors, 0.75 on input.
omega_n = 1 rad/s for outer loop.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solve_continuous_are

# ---- Vehicle parameters (Section V-A) ----
m = 0.50
g_vec = np.array([0.0, 0.0, -9.81])
gmag = 9.81
ITxx = 3.2e-3
ITyy = 3.2e-3
ITzz = 5.5e-3
IPxx = 0.0
IPzz = 1.5e-5
IBxx = ITxx - 4 * IPxx
IByy = IBxx
IBzz = ITzz - 4 * IPzz
l = 0.17
kf = 6.41e-6
ktau = 1.69e-2
gamma = 2.75e-3
sigma_mot = 0.015
F_MIN, F_MAX = 0.2, 3.8


def omega_signed(i, fi):
    s = -1.0 if i in (1, 3) else +1.0
    return s * np.sqrt(max(fi, 0.0) / kf)


# ============================================================
# 1. Equilibrium (closed-form, eq 22-25)
# ============================================================
def equilibrium_two_prop():
    f1 = m * gmag / 2
    f3 = f1
    f2 = 0.0
    f4 = 0.0
    om1 = -np.sqrt(m * gmag / (2 * kf))
    om3 = om1
    om2 = 0.0
    om4 = 0.0
    omegaSigma = om1 + om2 + om3 + om4
    r_bar = ktau * m * gmag / gamma
    omega_B = np.array([0.0, 0.0, r_bar])
    n_bar = np.array([0.0, 0.0, 1.0])
    f_Sigma = f1 + f3
    return dict(
        f=np.array([f1, f2, f3, f4]),
        omega_motors=np.array([om1, om2, om3, om4]),
        omegaSigma=omegaSigma,
        omega_B=omega_B,
        n_bar=n_bar,
        f_Sigma=f_Sigma,
    )


# ============================================================
# 2. Linearization (eq 27 with n_z=1, plus single-input B(2) eq 35)
# ============================================================
def linearize(eq):
    r_bar = eq["omega_B"][2]
    n_z = eq["n_bar"][2]  # = 1
    omegaSigma = eq["omegaSigma"]
    a_bar = (ITxx - ITzz) / IBxx * r_bar - IPzz / IBxx * omegaSigma  # eq (28)
    A = np.array(
        [
            [0.0, a_bar, 0.0, 0.0],
            [-a_bar, 0.0, 0.0, 0.0],
            [0.0, -n_z, 0.0, r_bar],
            [n_z, 0.0, -r_bar, 0.0],
        ]
    )
    # eq (35): single input on roll only (f3-f1 differential)
    B2 = (l / IBxx) * np.array([[0.0], [1.0], [0.0], [0.0]])
    return A, B2, a_bar


def extend_with_motors(A, B):
    n, m_in = B.shape
    Ae = np.block([[A, B], [np.zeros((m_in, n)), -np.eye(m_in) / sigma_mot]])
    Be = np.block([[np.zeros((n, m_in))], [np.eye(m_in) / sigma_mot]])
    return Ae, Be


def lqr(A, B, Q, R):
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.solve(R, B.T @ P)
    return K, P


# ============================================================
# 3. Nonlinear simulator (same eqs as 1-prop case)
# ============================================================
def hat(v):
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def rk4_step(f, x, dt, *args):
    k1 = f(x, *args)
    k2 = f(x + 0.5 * dt * k1, *args)
    k3 = f(x + 0.5 * dt * k2, *args)
    k4 = f(x + dt * k3, *args)
    return x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0


def dynamics(state, f_cmd):
    d = state[0:3]
    v = state[3:6]
    R = state[6:15].reshape(3, 3)
    om = state[15:18]
    p, q, r = om
    f = state[18:22]
    f1, f2, f3, f4 = f
    f_Sigma = f.sum()
    a = (R[:, 2] * f_Sigma) / m + g_vec
    dR = (R @ hat(om)).reshape(-1)
    om_motors = np.array(
        [
            omega_signed(1, f1),
            omega_signed(2, f2),
            omega_signed(3, f3),
            omega_signed(4, f4),
        ]
    )
    omSig = om_motors.sum()
    pdot = (((f2 - f4) * l) - (ITzz - ITxx) * q * r - IPzz * q * omSig) / IBxx
    qdot = (((f3 - f1) * l) + (ITzz - ITxx) * p * r + IPzz * p * omSig) / IByy
    rdot = (-gamma * r + ktau * (f1 - f2 + f3 - f4)) / IBzz
    df = (f_cmd - f) / sigma_mot
    return np.concatenate([v, a, dR, [pdot, qdot, rdot], df])


# ============================================================
# 4. Cascaded controller
# ============================================================
class Controller:
    def __init__(self, eq, K, omega_n=1.0, zeta=0.7):
        self.K = K
        self.omega_n = omega_n
        self.zeta = zeta
        self.fbar = eq["f"]
        self.n_bar = eq["n_bar"]
        self.omega_B_bar = eq["omega_B"]

    def translational(self, d, v, d_des):
        return -2 * self.zeta * self.omega_n * v - self.omega_n**2 * (d - d_des)

    def attitude_setpoint(self, R, a_des):
        # eq 45
        force_body = m * R.T @ (a_des - g_vec)
        norm_fb = np.linalg.norm(force_body)
        n_des_body = force_body / norm_fb
        # n_z_bar = 1, so f_Sigma_cmd = norm_fb / 1
        f_Sigma_cmd = norm_fb / self.n_bar[2]
        return n_des_body, f_Sigma_cmd

    def __call__(self, state, d_des):
        d = state[0:3]
        v = state[3:6]
        R = state[6:15].reshape(3, 3)
        om = state[15:18]
        f_actual = state[18:22]

        a_des = self.translational(d, v, d_des)
        n_des_body, f_Sigma_cmd = self.attitude_setpoint(R, a_des)
        n = n_des_body
        s_tilde = np.array(
            [
                om[0] - self.omega_B_bar[0],
                om[1] - self.omega_B_bar[1],
                n[0] - self.n_bar[0],
                n[1] - self.n_bar[1],
            ]
        )
        # extended state has 1 motor state: u1_actual = (f3-f3bar)-(f1-f1bar)
        f_dev = f_actual - self.fbar
        u1_actual = f_dev[2] - f_dev[0]
        x_ext = np.concatenate([s_tilde, [u1_actual]])
        u_cmd = -self.K @ x_ext
        u1 = u_cmd[0]

        # Map (u1, fSigma_cmd) to f1, f3 (eq 36 and definition of u1)
        # f1 + f3 = fSigma_cmd ; f3 - f1 = u1 + (f3bar - f1bar) = u1
        f1c = (f_Sigma_cmd - u1) / 2
        f3c = (f_Sigma_cmd + u1) / 2
        f_cmd = np.array([f1c, 0.0, f3c, 0.0])
        f_cmd[[0, 2]] = np.clip(f_cmd[[0, 2]], F_MIN, F_MAX)
        return f_cmd


# ============================================================
# 5. Run
# ============================================================
def run():
    eq = equilibrium_two_prop()
    print("=== Two-propeller equilibrium ===")
    print(f" f_bar     : {eq['f']}  (paper: f1=f3=2.45 N)")
    print(f" omega_mot : {eq['omega_motors']}  (paper: -619 rad/s)")
    print(f" omega_B   : {eq['omega_B']}  (paper: (0,0,30.1))")
    print(f" n_bar     : {eq['n_bar']}")
    print(f" f_Sigma   : {eq['f_Sigma']:.4f}  m g = {m * gmag:.4f}")

    A, B, a_bar = linearize(eq)
    print(f" a_bar     : {a_bar:.4f}")
    Cmat = np.hstack([B, A @ B, A @ A @ B, A @ A @ A @ B])
    print(f" rank C(2) : {np.linalg.matrix_rank(Cmat)} (need 4)")
    # Controllability conditions eq (38): a_bar*(a_bar+r_bar)^2 != 0
    r_bar = eq["omega_B"][2]
    print(f" a_bar*(a_bar+r_bar)^2 = {a_bar * (a_bar + r_bar) ** 2:.3f}")

    Ae, Be = extend_with_motors(A, B)
    # paper Section V-E weights
    Q = np.diag([0.0, 0.0, 1000.0, 2.0, 0.0])
    Q[0, 0] = 1e-6
    Q[1, 1] = 1e-6
    Q[4, 4] = 1e-6  # PSD numerical
    R = np.array([[0.75]])
    K, _ = lqr(Ae, Be, Q, R)
    print(f" K = {K}")

    ctrl = Controller(eq, K, omega_n=1.0, zeta=0.7)

    dt = 1 / 1000
    T_end = 25.0
    N = int(T_end / dt)
    R0 = np.eye(3)
    # Start on ground at rest (paper Fig.5)
    state = np.concatenate(
        [
            np.array([0, 0, 0.0]),
            np.zeros(3),
            R0.reshape(-1),
            np.zeros(3),
            np.zeros(4),
        ]
    )

    # Setpoint per Fig.5: hover at z=2 m. After 5.6 s shift x by 1 m.
    def setpoint(t):
        d = np.array([0, 0, 2.0])
        if t >= 5.6:
            d = np.array([1, 0, 2.0])
        return d

    # Spin-up: only props 1,3 at full thrust until |omega| > 10 rad/s
    spin_thrust = eq["f"][0]  # 2.45 N
    pre_phase_cmd = np.array([spin_thrust, 0.0, spin_thrust, 0.0])

    log_t = np.zeros(N + 1)
    log_d = np.zeros((N + 1, 3))
    log_v = np.zeros((N + 1, 3))
    log_om = np.zeros((N + 1, 3))
    log_f = np.zeros((N + 1, 4))
    log_n = np.zeros((N + 1, 3))
    log_d_des = np.zeros((N + 1, 3))
    enabled_t = None

    s = state.copy()
    for k in range(N + 1):
        t = k * dt
        Rm = s[6:15].reshape(3, 3)
        om_now = s[15:18]
        n_body = Rm.T @ np.array([0, 0, 1.0])
        log_t[k] = t
        log_d[k] = s[0:3]
        log_v[k] = s[3:6]
        log_om[k] = om_now
        log_f[k] = s[18:22]
        log_n[k] = n_body
        d_des = setpoint(t)
        log_d_des[k] = d_des
        if k == N:
            break
        if enabled_t is None and np.linalg.norm(om_now) > 10.0:
            enabled_t = t
            print(f" LQR enabled at t = {t:.3f} s")
        if enabled_t is None:
            f_cmd = pre_phase_cmd.copy()
        else:
            f_cmd = ctrl(s, d_des)
        s = rk4_step(dynamics, s, dt, f_cmd)

    print(f" Final pos   : {s[0:3]}")
    print(f" Final omega : {s[15:18]}")
    print(f" Final f     : {s[18:22]}")

    plot_results(log_t, log_d, log_d_des, log_om, log_f, log_n, eq, enabled_t)


def plot_results(t, d, d_des, om, f, n, eq, enabled_t):
    fig, axs = plt.subplots(5, 1, figsize=(9, 12), sharex=True)
    axs[0].plot(t, d[:, 0], label="x")
    axs[0].plot(t, d_des[:, 0], "--", label="x_des")
    axs[0].plot(t, d[:, 1], label="y")
    axs[0].plot(t, d_des[:, 1], "--", label="y_des")
    axs[0].set_ylabel("horizontal [m]")
    axs[0].legend()
    axs[0].grid(True)
    axs[0].set_title("Fig. 5 replication: 2 opposing-prop failure")
    axs[1].plot(t, d[:, 2], label="z")
    axs[1].plot(t, d_des[:, 2], "--", label="z_des")
    axs[1].set_ylabel("height [m]")
    axs[1].legend()
    axs[1].grid(True)
    axs[2].plot(t, om[:, 0], label="p")
    axs[2].plot(t, om[:, 1], label="q")
    axs[2].plot(t, om[:, 2], label="r")
    for v_, c in zip(eq["omega_B"], ["C0", "C1", "C2"]):
        axs[2].axhline(v_, ls=":", color=c, alpha=0.6)
    axs[2].set_ylabel("omega_B [rad/s]")
    axs[2].legend()
    axs[2].grid(True)
    axs[3].plot(t, f[:, 0], label="f1")
    axs[3].plot(t, f[:, 1], label="f2")
    axs[3].plot(t, f[:, 2], label="f3")
    axs[3].plot(t, f[:, 3], label="f4")
    axs[3].axhline(F_MAX, ls=":", color="k")
    axs[3].axhline(F_MIN, ls=":", color="k")
    axs[3].set_ylabel("Forces [N]")
    axs[3].legend()
    axs[3].grid(True)
    axs[4].plot(t, n[:, 0], label="n_x")
    axs[4].plot(t, n[:, 1], label="n_y")
    axs[4].plot(t, n[:, 2], label="n_z")
    axs[4].set_ylabel("inertial-up in body")
    axs[4].legend()
    axs[4].grid(True)
    axs[4].set_xlabel("time [s]")
    if enabled_t is not None:
        for ax in axs:
            ax.axvline(enabled_t, ls="--", color="gray", alpha=0.5)
    plt.tight_layout()
    plt.savefig("/Users/dikshant/Desktop/uav-project/fig5_two_prop.png", dpi=130)
    print(" Saved fig5_two_prop.png")


if __name__ == "__main__":
    run()
