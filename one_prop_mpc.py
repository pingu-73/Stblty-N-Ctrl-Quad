"""
Alternative controller for the 1-propeller-failure case from
Mueller & D'Andrea (ICRA 2014):

  Replaces the inner-loop LQR (paper Section V-D) with a linear
  Model Predictive Controller (MPC) on the same extended reduced-attitude
  model (state + motor first-order dynamics).

Why MPC?
  - The paper notes thrust saturation f_i in [0.2, 3.8] N is binding.
    LQR ignores constraints; an MPC formulates them as a QP and respects
    them in the prediction horizon, not just by post-hoc clipping.
  - Same physical model as the paper: discretized A_e, B_e from
    one_prop_failure.py.

Outer translational loop is unchanged (eq 44–45, omega_n=1, zeta=0.7).
"""
import numpy as np
import osqp
from scipy import sparse
from scipy.linalg import expm, solve_discrete_are
import matplotlib.pyplot as plt

# Reuse model + equilibrium from the LQR script
from one_prop_failure import (
    m, gmag, g_vec, ITxx, ITzz, IBxx, IPzz, l, kf, ktau, gamma,
    sigma_mot, F_MIN, F_MAX,
    omega_signed, equilibrium_one_prop, linearize, extend_with_motors,
    dynamics, rk4_step, hat,
    Controller as LQRController, lqr,
)

# ============================================================
# Discretize continuous extended system (A_e, B_e) via ZOH
# ============================================================
def discretize_zoh(A, B, dt):
    n, m_in = B.shape
    M = np.zeros((n + m_in, n + m_in))
    M[:n, :n] = A
    M[:n, n:] = B
    Md = expm(M * dt)
    Ad = Md[:n, :n]
    Bd = Md[:n, n:]
    return Ad, Bd


# ============================================================
# Linear MPC for the extended attitude system
#
#   state z_k = [s_tilde; u_actual]  (6-dim)
#   inputs v_k = (u1_cmd, u2_cmd)    (2-dim)
#   dynamics: z_{k+1} = Ad z_k + Bd v_k
#   stage cost: z' Q z + v' R v
#   terminal: z' P_N z   (P_N from DARE)
#   box on motor commands: depends on commanded fSigma → built per step
# ============================================================
class MPCController:
    def __init__(self, eq, Ad, Bd, Q, R, N, omega_n=1.0, zeta=0.7):
        self.eq = eq
        self.fbar = eq['f']
        self.n_bar = eq['n_bar']
        self.omega_B_bar = eq['omega_B']
        self.fSigma_bar = eq['f_Sigma']
        self.Ad = Ad
        self.Bd = Bd
        self.Q = Q
        self.R = R
        self.N = N
        self.omega_n = omega_n
        self.zeta = zeta

        nx, nu = Bd.shape
        self.nx = nx
        self.nu = nu

        # Terminal cost via DARE
        self.PN = solve_discrete_are(Ad, Bd, Q, R)

        # Build sparse OSQP problem (state + input decision vars):
        #   z = [x_0; x_1; ...; x_N; u_0; ...; u_{N-1}]
        # Equality constraints:
        #   x_0 = x_init (set per step)
        #   x_{k+1} = Ad x_k + Bd u_k
        # Inequality constraints:
        #   per-step linear bounds on u_k (motor box; updated per step)
        self._setup_osqp()

    def _setup_osqp(self):
        nx, nu, N = self.nx, self.nu, self.N
        # Cost Hessian
        # blocks: Q for x_0..x_{N-1}, P_N for x_N, R for u_0..u_{N-1}
        H_x = sparse.block_diag([sparse.csc_matrix(self.Q)] * N
                                + [sparse.csc_matrix(self.PN)])
        H_u = sparse.block_diag([sparse.csc_matrix(self.R)] * N)
        P = sparse.block_diag([H_x, H_u]).tocsc()
        self.P = P
        self.q = np.zeros(P.shape[0])

        # Equality dynamics:  -x_{k+1} + Ad x_k + Bd u_k = 0
        # plus initial state equality: x_0 = x_init
        n_dyn = N * nx + nx  # initial constraint + N dynamics rows
        n_x_total = (N + 1) * nx
        n_u_total = N * nu
        # x part of A_eq
        Ax = sparse.lil_matrix((n_dyn, n_x_total))
        Au = sparse.lil_matrix((n_dyn, n_u_total))
        # initial state: x_0 = x_init (rows 0..nx-1)
        Ax[0:nx, 0:nx] = sparse.eye(nx)
        # dynamics rows nx..nx+N*nx
        for k in range(N):
            r0 = nx + k * nx
            # -x_{k+1}
            Ax[r0:r0 + nx, (k + 1) * nx:(k + 2) * nx] = -sparse.eye(nx)
            # +Ad x_k
            Ax[r0:r0 + nx, k * nx:(k + 1) * nx] = sparse.csc_matrix(self.Ad)
            # +Bd u_k
            Au[r0:r0 + nx, k * nu:(k + 1) * nu] = sparse.csc_matrix(self.Bd)
        A_eq = sparse.hstack([Ax, Au]).tocsc()

        # Inequality on inputs: motor box constraints, 6 rows per step
        # f_i in [F_MIN, F_MAX] with
        #   f1 = (fSigma_cmd - f2bar - u2 - u1)/2
        #   f2 = f2bar + u2
        #   f3 = (fSigma_cmd - f2bar - u2 + u1)/2
        # Rearranged as 3 linear functions of (u1,u2):
        #   f1 contribution: -u1/2 - u2/2 + (fSigma_cmd - f2bar)/2
        #   f2 contribution:  u2 + f2bar
        #   f3 contribution:  u1/2 - u2/2 + (fSigma_cmd - f2bar)/2
        # We'll set bounds: F_MIN <= ... <= F_MAX.
        # Coefficients on u_k:
        Gu = np.array([
            [-0.5, -0.5],   # f1 / (also has fSigma offset)
            [ 0.0,  1.0],   # f2 / (also offset f2bar)
            [ 0.5, -0.5],   # f3 / (also has fSigma offset)
        ])
        rows_per_step = Gu.shape[0]
        n_ineq = N * rows_per_step
        Aineq_u = sparse.lil_matrix((n_ineq, n_u_total))
        for k in range(N):
            Aineq_u[k * rows_per_step:(k + 1) * rows_per_step,
                    k * nu:(k + 1) * nu] = sparse.csc_matrix(Gu)
        Aineq_x = sparse.lil_matrix((n_ineq, n_x_total))
        A_ineq = sparse.hstack([Aineq_x, Aineq_u]).tocsc()

        self.Gu = Gu
        self.A_constr = sparse.vstack([A_eq, A_ineq]).tocsc()

        # Lower / upper placeholders. Equalities: l=u=rhs.
        self.l = np.zeros(self.A_constr.shape[0])
        self.u = np.zeros(self.A_constr.shape[0])
        # No initial values — will be filled per step.

        self.solver = osqp.OSQP()
        self.solver.setup(self.P, self.q, self.A_constr, self.l, self.u,
                          warm_starting=True, verbose=False, polishing=True,
                          eps_abs=1e-5, eps_rel=1e-5)

        # Cache offsets for updates
        self._n_eq = (N + 1) * nx  # initial + dynamics rows total
        self._n_x_total = n_x_total

    def _update_bounds(self, x_init, fSigma_cmd):
        nx, nu, N = self.nx, self.nu, self.N
        # Equality RHS: initial state then 0 for dynamics
        rhs_eq = np.zeros(self._n_eq)
        rhs_eq[0:nx] = x_init  # x_0 = x_init
        # Dynamics rows have RHS = 0 (since -x_{k+1}+Ad x_k+Bd u_k = 0)

        # Inequality bounds: per-step
        # f1: F_MIN <= -u1/2 - u2/2 + (fSigma - f2bar)/2 <= F_MAX
        # f2: F_MIN <= u2 + f2bar <= F_MAX
        # f3: F_MIN <= u1/2 - u2/2 + (fSigma - f2bar)/2 <= F_MAX
        f2bar = self.fbar[1]
        c_f13 = (fSigma_cmd - f2bar) / 2.0
        c_f2 = f2bar
        # Per-step lower/upper after subtracting offset
        l_step = np.array([F_MIN - c_f13, F_MIN - c_f2, F_MIN - c_f13])
        u_step = np.array([F_MAX - c_f13, F_MAX - c_f2, F_MAX - c_f13])
        l_ineq = np.tile(l_step, N)
        u_ineq = np.tile(u_step, N)

        self.l = np.concatenate([rhs_eq, l_ineq])
        self.u = np.concatenate([rhs_eq, u_ineq])
        self.solver.update(l=self.l, u=self.u)

    # Translational outer loop (same as LQR)
    def translational(self, d, v, d_des):
        return -2 * self.zeta * self.omega_n * v - self.omega_n**2 * (d - d_des)

    def attitude_setpoint(self, R, a_des):
        force_body = m * R.T @ (a_des - g_vec)
        norm_fb = np.linalg.norm(force_body)
        n_des_body = force_body / norm_fb
        f_Sigma_cmd = norm_fb / self.n_bar[2]
        return n_des_body, f_Sigma_cmd

    def __call__(self, state, d_des):
        d = state[0:3]; v = state[3:6]
        Rmat = state[6:15].reshape(3, 3)
        om = state[15:18]
        f_actual = state[18:22]

        a_des = self.translational(d, v, d_des)
        n_des_body, fSigma_cmd = self.attitude_setpoint(Rmat, a_des)
        n = n_des_body

        # Build initial extended state (6-D) in deviation form
        s_tilde = np.array([
            om[0] - self.omega_B_bar[0],
            om[1] - self.omega_B_bar[1],
            n[0] - self.n_bar[0],
            n[1] - self.n_bar[1],
        ])
        f_dev = f_actual[:3] - self.fbar[:3]
        u1_actual = f_dev[2] - f_dev[0]
        u2_actual = f_dev[1]
        x_init = np.concatenate([s_tilde, [u1_actual, u2_actual]])

        self._update_bounds(x_init, fSigma_cmd)
        res = self.solver.solve()
        if res.info.status not in ("solved", "solved inaccurate"):
            # fallback: zero input deviation
            u1, u2 = 0.0, 0.0
        else:
            # First control input
            offset = self._n_x_total  # u_0 starts after all x's
            u1 = res.x[offset + 0]
            u2 = res.x[offset + 1]

        # Map to motor commands
        f2c = self.fbar[1] + u2
        rest = fSigma_cmd - f2c
        f1c = (rest - u1) / 2.0
        f3c = (rest + u1) / 2.0
        f_cmd = np.array([f1c, f2c, f3c, 0.0])
        f_cmd[:3] = np.clip(f_cmd[:3], F_MIN, F_MAX)
        return f_cmd


# ============================================================
# Run + compare vs LQR
# ============================================================
def run():
    rho = 0.5
    eq = equilibrium_one_prop(rho)
    A, B, a_bar = linearize(eq)
    Ae, Be = extend_with_motors(A, B)

    # Discretize at controller rate
    dt_c = 0.01            # 100 Hz inner
    N_h  = 20              # 0.2 s horizon
    Ad, Bd = discretize_zoh(Ae, Be, dt_c)

    # Same weights as paper LQR (Section V-D)
    Q = np.diag([1.0, 1.0, 20.0, 20.0, 1e-6, 1e-6])
    R = np.diag([1.0, 1.0])

    mpc = MPCController(eq, Ad, Bd, Q, R, N_h)

    # LQR for comparison (continuous-time, same Q, R)
    K, _ = lqr(Ae, Be, Q, R)
    lqr_ctrl = LQRController(eq, K, omega_n=1.0, zeta=0.7)

    # Setpoint schedule (Fig. 4 of paper)
    def setpoint(t):
        d = np.array([0, 0, 2.0])
        if t >= 7.7:  d = np.array([1, 0, 2.0])
        if t >= 19.8: d = np.array([1, 0, 0.0])
        return d

    # Run a single sim with chosen controller
    def simulate(controller, label):
        dt = 1.0 / 1000
        T_end = 25.0
        N = int(T_end / dt)
        ctrl_period = 0.01  # apply controller every 10 ms
        ctrl_steps  = int(round(ctrl_period / dt))

        R0 = np.eye(3)
        f_hover = (m * gmag / 4) * np.ones(4)
        state = np.concatenate([
            np.array([0, 0, 2.0]), np.zeros(3),
            R0.reshape(-1), np.zeros(3), f_hover,
        ])
        spin_thrust = 2.05
        pre_phase_cmd = np.array([spin_thrust, 0.0, spin_thrust, 0.0])

        log_t   = np.zeros(N + 1)
        log_d   = np.zeros((N + 1, 3))
        log_om  = np.zeros((N + 1, 3))
        log_f   = np.zeros((N + 1, 4))
        log_n   = np.zeros((N + 1, 3))
        log_dd  = np.zeros((N + 1, 3))
        enabled_t = None
        f_cmd = pre_phase_cmd.copy()

        s = state.copy()
        for k in range(N + 1):
            t = k * dt
            Rm = s[6:15].reshape(3, 3)
            om_now = s[15:18]
            n_body = Rm.T @ np.array([0, 0, 1.0])
            log_t[k] = t; log_d[k] = s[0:3]; log_om[k] = om_now
            log_f[k] = s[18:22]; log_n[k] = n_body; log_dd[k] = setpoint(t)
            if k == N: break
            if enabled_t is None and np.linalg.norm(om_now) > 10.0:
                enabled_t = t
                print(f"  [{label}] LQR/MPC engaged at t={t:.3f} s")
            if enabled_t is None:
                f_cmd = pre_phase_cmd.copy()
            elif k % ctrl_steps == 0:
                out = controller(s, setpoint(t))
                f_cmd = out[0] if isinstance(out, tuple) else out
            s = rk4_step(dynamics, s, dt, f_cmd)
        return log_t, log_d, log_om, log_f, log_n, log_dd, enabled_t

    print("Running LQR sim ...")
    t_l, d_l, om_l, f_l, n_l, dd_l, et_l = simulate(lqr_ctrl, "LQR")
    print("Running MPC sim ...")
    t_m, d_m, om_m, f_m, n_m, dd_m, et_m = simulate(mpc, "MPC")

    # Plot comparison
    fig, axs = plt.subplots(5, 1, figsize=(10, 13), sharex=True)

    axs[0].plot(t_l, d_l[:, 0], label='LQR x', color='C0')
    axs[0].plot(t_m, d_m[:, 0], label='MPC x', color='C0', ls='--')
    axs[0].plot(t_l, d_l[:, 1], label='LQR y', color='C1')
    axs[0].plot(t_m, d_m[:, 1], label='MPC y', color='C1', ls='--')
    axs[0].plot(t_l, dd_l[:, 0], ':k', label='x_des')
    axs[0].set_ylabel('horizontal [m]'); axs[0].legend(ncol=3); axs[0].grid(True)
    axs[0].set_title('1-prop failure: LQR (solid) vs MPC (dashed)')

    axs[1].plot(t_l, d_l[:, 2], label='LQR z', color='C2')
    axs[1].plot(t_m, d_m[:, 2], label='MPC z', color='C2', ls='--')
    axs[1].plot(t_l, dd_l[:, 2], ':k', label='z_des')
    axs[1].set_ylabel('height [m]'); axs[1].legend(); axs[1].grid(True)

    axs[2].plot(t_l, om_l[:, 1], label='LQR q')
    axs[2].plot(t_m, om_m[:, 1], label='MPC q', ls='--')
    axs[2].plot(t_l, om_l[:, 2], label='LQR r')
    axs[2].plot(t_m, om_m[:, 2], label='MPC r', ls='--')
    for v_, c in zip(eq['omega_B'][1:], ['C0', 'C1']):
        axs[2].axhline(v_, ls=':', color=c, alpha=0.5)
    axs[2].set_ylabel('omega_B [rad/s]'); axs[2].legend(); axs[2].grid(True)

    axs[3].plot(t_l, f_l[:, 0], label='LQR f1')
    axs[3].plot(t_m, f_m[:, 0], label='MPC f1', ls='--')
    axs[3].plot(t_l, f_l[:, 1], label='LQR f2')
    axs[3].plot(t_m, f_m[:, 1], label='MPC f2', ls='--')
    axs[3].plot(t_l, f_l[:, 2], label='LQR f3')
    axs[3].plot(t_m, f_m[:, 2], label='MPC f3', ls='--')
    axs[3].axhline(F_MAX, ls=':', color='k'); axs[3].axhline(F_MIN, ls=':', color='k')
    axs[3].set_ylabel('Forces [N]'); axs[3].legend(ncol=3); axs[3].grid(True)

    axs[4].plot(t_l, n_l[:, 0], label='LQR n_x')
    axs[4].plot(t_m, n_m[:, 0], label='MPC n_x', ls='--')
    axs[4].plot(t_l, n_l[:, 1], label='LQR n_y')
    axs[4].plot(t_m, n_m[:, 1], label='MPC n_y', ls='--')
    axs[4].set_ylabel('inertial-up in body'); axs[4].legend(); axs[4].grid(True)
    axs[4].set_xlabel('time [s]')
    plt.tight_layout()
    plt.savefig('/Users/dikshant/Desktop/uav-project/fig_mpc_vs_lqr.png', dpi=130)
    print(" Saved fig_mpc_vs_lqr.png")

    # Summary metrics: RMS x-error post engage, peak overshoot, control effort
    def metrics(t, d, dd, f, et):
        mask = t >= (et + 0.5 if et else 0.0)
        ex = d[mask, 0] - dd[mask, 0]
        ey = d[mask, 1] - dd[mask, 1]
        ez = d[mask, 2] - dd[mask, 2]
        rms = np.sqrt(np.mean(ex**2 + ey**2 + ez**2))
        # control effort approximated as L2 of motor forces over time
        u = np.sum(np.abs(f[mask, :3]), axis=1)
        return rms, np.mean(u)
    rms_l, eff_l = metrics(t_l, d_l, dd_l, f_l, et_l)
    rms_m, eff_m = metrics(t_m, d_m, dd_m, f_m, et_m)
    print(f"\n=== Performance comparison ===")
    print(f" LQR : pos RMSE = {rms_l:.4f} m   mean |f| sum = {eff_l:.3f} N")
    print(f" MPC : pos RMSE = {rms_m:.4f} m   mean |f| sum = {eff_m:.3f} N")


if __name__ == '__main__':
    run()
