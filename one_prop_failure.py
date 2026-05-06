"""
- Solves periodic equilibrium for chosen rho = f2/f1
- Linearizes reduced attitude (p, q, nx, ny) plus motor dynamics extension
- Designs cascaded controller: LQR inner loop (attitude), 2nd-order outer (translation)
- Simulates the full nonlinear quadrocopter with propeller 4 disabled
"""
import numpy as np
from scipy.linalg import solve_continuous_are
from scipy.optimize import brentq
import matplotlib.pyplot as plt

# ---------- Vehicle parameters (Section V-A) ----------
m      = 0.50                  # kg
g_vec  = np.array([0.0, 0.0, -9.81])
gmag   = 9.81
ITxx   = 3.2e-3                # = ITyy (symmetric)
ITyy   = 3.2e-3
ITzz   = 5.5e-3
IPxx   = 0.0                   # neglected
IPzz   = 1.5e-5
IBxx   = ITxx - 4*IPxx         # = ITxx since IPxx=0
IByy   = IBxx
IBzz   = ITzz - 4*IPzz
l      = 0.17                  # m
kf     = 6.41e-6               # N s^2/rad^2
ktau   = 1.69e-2               # N m / N
gamma  = 2.75e-3               # N m s/rad
sigma_mot = 0.015              # s
F_MIN, F_MAX = 0.2, 3.8        # N

# Sign convention: propellers 1,3 spin negative (omega<0), 2,4 spin positive.
# tau_i = (-1)^(i+1) ktau f_i  =>  yaw torque sum = ktau (f1 - f2 + f3 - f4).
def omega_signed(i, fi):
    s = -1.0 if i in (1, 3) else +1.0
    return s * np.sqrt(max(fi, 0.0) / kf)

# ============================================================
# 1. Periodic equilibrium for one-propeller failure (rho given)
# ============================================================
# Propeller 4 dead: omega4 = f4 = 0. Choose f1=f3, rho=f2/f1.
# Symmetry => p_bar = 0, n_bar_x = 0.
# Yaw eq:    gamma r_bar = ktau (2 - rho) f1
# Pitch eq:  rho f1 l = q_bar [(ITzz-ITxx) r_bar + IPzz omegaSigma]
# Constraint f_Sigma * n_z = m g, with n = omega_B / |omega_B|.

def equilibrium_one_prop(rho):
    def residual(f1):
        f2 = rho * f1
        f3 = f1
        omegaSigma = (omega_signed(1, f1) + omega_signed(2, f2) +
                      omega_signed(3, f3) + 0.0)
        r_bar = ktau * (2.0 - rho) * f1 / gamma
        denom = (ITzz - ITxx) * r_bar + IPzz * omegaSigma
        q_bar = rho * f1 * l / denom
        omega_norm = np.sqrt(q_bar**2 + r_bar**2)
        n_z = r_bar / omega_norm
        f_Sigma = (2.0 + rho) * f1
        return f_Sigma * n_z - m * gmag

    f1 = brentq(residual, 0.05, 5.0)
    f2 = rho * f1
    f3 = f1
    f4 = 0.0
    om1 = omega_signed(1, f1); om2 = omega_signed(2, f2)
    om3 = omega_signed(3, f3); om4 = 0.0
    omegaSigma = om1 + om2 + om3 + om4
    r_bar = ktau * (2.0 - rho) * f1 / gamma
    q_bar = rho * f1 * l / ((ITzz - ITxx) * r_bar + IPzz * omegaSigma)
    p_bar = 0.0
    omega_B = np.array([p_bar, q_bar, r_bar])
    om_norm = np.linalg.norm(omega_B)
    n_bar = omega_B / om_norm
    eps   = 1.0 / om_norm
    f_Sigma = f1 + f2 + f3 + f4
    R_ps  = np.sqrt(1 - n_bar[2]**2) / n_bar[2] * gmag / om_norm**2
    T_ps  = 2*np.pi / om_norm
    return dict(f=np.array([f1,f2,f3,f4]),
                omega_motors=np.array([om1,om2,om3,om4]),
                omegaSigma=omegaSigma,
                omega_B=omega_B, n_bar=n_bar, eps=eps,
                f_Sigma=f_Sigma, R_ps=R_ps, T_ps=T_ps)

# ============================================================
# 2. Linearization (eq 27, 33) + motor extension (eq 47-49)
# ============================================================
def linearize(eq):
    r_bar = eq['omega_B'][2]
    n_z   = eq['n_bar'][2]
    omegaSigma = eq['omegaSigma']
    a_bar = (ITxx - ITzz)/IBxx * r_bar - IPzz/IBxx * omegaSigma  # eq (28)
    A = np.array([[0.0,  a_bar, 0.0,    0.0],
                  [-a_bar, 0.0, 0.0,    0.0],
                  [0.0,  -n_z, 0.0,   r_bar],
                  [n_z,   0.0, -r_bar,  0.0]])
    # B(3): inputs u1 = (f3-f3bar) - (f1-f1bar),  u2 = f2-f2bar
    B3 = (l/IBxx) * np.array([[0.0, 1.0],
                              [1.0, 0.0],
                              [0.0, 0.0],
                              [0.0, 0.0]])
    return A, B3, a_bar

def extend_with_motors(A, B):
    # Extended state: s_e = [s; f_dev]   where f_dev tracks u as 1st order with sigma_mot
    # ds/dt = A s + B f_dev ;  d f_dev/dt = (u_cmd - f_dev)/sigma_mot
    n, m_in = B.shape
    Ae = np.block([[A,                  B],
                   [np.zeros((m_in,n)), -np.eye(m_in)/sigma_mot]])
    Be = np.block([[np.zeros((n, m_in))],
                   [np.eye(m_in)/sigma_mot]])
    return Ae, Be

# ============================================================
# 3. LQR controller design
# ============================================================
def lqr(A, B, Q, R):
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.solve(R, B.T @ P)
    return K, P

# ============================================================
# 4. Nonlinear simulator
# ============================================================
def hat(v):
    return np.array([[0.0, -v[2], v[1]],
                     [v[2], 0.0, -v[0]],
                     [-v[1], v[0], 0.0]])

def rk4_step(f, x, dt, *args):
    k1 = f(x,                *args)
    k2 = f(x + 0.5*dt*k1,    *args)
    k3 = f(x + 0.5*dt*k2,    *args)
    k4 = f(x + dt*k3,        *args)
    return x + dt*(k1 + 2*k2 + 2*k3 + k4)/6.0

def dynamics(state, f_cmd):
    """
    state layout:
      0:3   d  (position, inertial)
      3:6   v  (velocity, inertial)
      6:15  R  (rotation body->inertial, row-major flatten)
      15:18 omega_B = (p,q,r)
      18:22 f_actual (forces, 4 props)
    f_cmd: commanded forces (4)
    """
    d = state[0:3]; v = state[3:6]
    R = state[6:15].reshape(3,3)
    om = state[15:18]; p, q, r = om
    f = state[18:22]
    f1, f2, f3, f4 = f

    # Translational: m d_ddot = R z f_Sigma + m g  (eq 1)
    f_Sigma = f.sum()
    z_b = R[:, 2]
    a = (z_b * f_Sigma) / m + g_vec
    dd = v
    dv = a

    # Rotation kinematics
    dR = (R @ hat(om)).reshape(-1)

    # Motor speeds (signed) from forces
    om_motors = np.array([omega_signed(1, f1), omega_signed(2, f2),
                          omega_signed(3, f3), omega_signed(4, f4)])
    omSig = om_motors.sum()

    # Body angular dynamics (eq 12-14)
    dp = (kf*(f2 - f4)/kf * 0.0)  # placeholder, recompute below cleanly
    # eq 12:  IBxx * pdot = kf*(om2^2-om4^2)*l - (ITzz-ITxx)*q*r - IPzz*q*omSig
    pdot = (((f2 - f4) * l) - (ITzz - ITxx) * q * r - IPzz * q * omSig) / IBxx
    qdot = (((f3 - f1) * l) + (ITzz - ITxx) * p * r + IPzz * p * omSig) / IByy
    rdot = (-gamma * r + ktau * (f1 - f2 + f3 - f4)) / IBzz
    dom = np.array([pdot, qdot, rdot])

    # Motor dynamics: 1st order to commanded
    df = (f_cmd - f) / sigma_mot

    return np.concatenate([dd, dv, dR, dom, df])

# ============================================================
# 5. Cascaded controller
# ============================================================
class Controller:
    def __init__(self, eq, K, omega_n=1.0, zeta=0.7):
        self.eq = eq
        self.K  = K
        self.omega_n = omega_n
        self.zeta = zeta
        self.fbar  = eq['f']
        self.fSigma_bar = eq['f_Sigma']
        self.n_bar = eq['n_bar']
        self.omega_B_bar = eq['omega_B']

    def translational(self, d, v, d_des):
        # eq 44: ddot_des + 2 zeta wn (v) + wn^2 (d - d_des) = 0
        a_des = -2*self.zeta*self.omega_n*v - self.omega_n**2*(d - d_des)
        return a_des

    def attitude_setpoint(self, R, a_des):
        # eq 45: required body force vector along desired primary axis
        # m R^{-1} (a_des - g) gives body-frame force vector
        force_body = m * R.T @ (a_des - g_vec)
        # Desired primary axis points along this force vector (in body frame)
        norm_fb = np.linalg.norm(force_body)
        n_des_body = force_body / norm_fb
        # Total thrust scalar: f_Sigma = norm / n_z_bar (per eq 45)
        f_Sigma_cmd = norm_fb / self.n_bar[2]
        return n_des_body, f_Sigma_cmd

    def __call__(self, state, d_des):
        d = state[0:3]; v = state[3:6]
        R = state[6:15].reshape(3,3)
        om = state[15:18]
        f_actual = state[18:22]

        a_des = self.translational(d, v, d_des)
        n_des_body, f_Sigma_cmd = self.attitude_setpoint(R, a_des)

        # Reduced attitude error = body-frame axis error from equilibrium
        # The "primary axis" (body-fixed) at equilibrium has body coords n_bar.
        # We want vehicle so its body z-ish axis n_bar points at n_des_inertial.
        # In body frame, the inertial "up" direction the vehicle should treat as
        # equilibrium-up is n_des_body. So state s = (p,q,nx,ny) with n = n_des_body
        # gives error s_tilde = s - s_bar.
        n = n_des_body  # body-frame components
        # error vector
        s_tilde = np.array([om[0] - self.omega_B_bar[0],
                            om[1] - self.omega_B_bar[1],
                            n[0]  - self.n_bar[0],
                            n[1]  - self.n_bar[1]])
        f_dev = f_actual[:3] - self.fbar[:3]   # only first 3 motors live
        # Note: extended LQR designed on (s_tilde, u-ubar) where u = (u1,u2)
        # u1 = (f3-f3bar)-(f1-f1bar), u2 = f2-f2bar
        u1_actual = f_dev[2] - f_dev[0]
        u2_actual = f_dev[1]
        x_ext = np.concatenate([s_tilde, [u1_actual, u2_actual]])
        u_cmd = -self.K @ x_ext     # commanded (u1,u2) deviation
        u1, u2 = u_cmd

        # Map (u1,u2,f_Sigma_cmd) -> commanded f1,f2,f3 (eq 29-31)
        f2c = self.fbar[1] + u2
        rest = f_Sigma_cmd - f2c
        f1c = (rest - u1) / 2.0
        f3c = (rest + u1) / 2.0

        # Saturate
        f_cmd = np.array([f1c, f2c, f3c, 0.0])
        f_cmd[:3] = np.clip(f_cmd[:3], F_MIN, F_MAX)
        return f_cmd, dict(a_des=a_des, n_des_body=n_des_body,
                           f_Sigma_cmd=f_Sigma_cmd, u=u_cmd)

# ============================================================
# 6. Run
# ============================================================
def run():
    rho = 0.5
    eq = equilibrium_one_prop(rho)
    print("=== Equilibrium (rho=0.5) ===")
    print(f" f_bar [N]      : {eq['f']}")
    print(f" omega_B [rad/s]: {eq['omega_B']}   |omega|={np.linalg.norm(eq['omega_B']):.3f}")
    print(f" n_bar          : {eq['n_bar']}")
    print(f" R_ps [m]       : {eq['R_ps']:.5f}")
    print(f" T_ps [s]       : {eq['T_ps']:.5f}")
    print(f" f_Sigma [N]    : {eq['f_Sigma']:.4f}  (m g = {m*gmag:.4f})")

    A, B, a_bar = linearize(eq)
    print(f" coupling a_bar = {a_bar:.4f}")
    # Controllability check (eq 33)
    Cmat = np.hstack([B, A@B, A@A@B, A@A@A@B])
    print(f" rank(C(3)) = {np.linalg.matrix_rank(Cmat)}  (need 4)")

    Ae, Be = extend_with_motors(A, B)
    # Q diag: 1 on (p,q), 20 on (nx,ny), 0 on motor states (use tiny eps for psd)
    Q = np.diag([1.0, 1.0, 20.0, 20.0, 1e-6, 1e-6])
    R = np.diag([1.0, 1.0])
    K, _ = lqr(Ae, Be, Q, R)
    print(f" LQR gain K shape {K.shape}")

    ctrl = Controller(eq, K, omega_n=1.0, zeta=0.7)

    # ----- Simulation -----
    dt = 1.0/1000  # 1 kHz inner
    T_end = 25.0
    N = int(T_end/dt)

    # Initial state: hover at z=2 m, all motors at hover thrust
    R0 = np.eye(3)
    f_hover = (m*gmag/4) * np.ones(4)
    state = np.concatenate([
        np.array([0,0,2.0]),         # d
        np.zeros(3),                  # v
        R0.reshape(-1),
        np.zeros(3),                  # omega
        f_hover,                      # forces
    ])

    # Setpoint schedule (per Fig.4): start at (0,0,2). At t=7.7s shift x by 1m. At 19.8s set z=0.
    def setpoint(t):
        d = np.array([0,0,2.0])
        if t >= 7.7:  d = np.array([1,0,2.0])
        if t >= 19.8: d = np.array([1,0,0.0])
        return d

    # Simulator runs full time, but the LQR controller is only enabled
    # once |omega| > 10 rad/s (per paper). Until then: only props 1 & 3 at hover-ish
    # to spin up about z (paper's strategy).
    spin_thrust = 2.05  # close to equilibrium f1
    pre_phase_cmd = np.array([spin_thrust, 0.0, spin_thrust, 0.0])

    log_t   = np.zeros(N+1)
    log_d   = np.zeros((N+1, 3))
    log_v   = np.zeros((N+1, 3))
    log_om  = np.zeros((N+1, 3))
    log_f   = np.zeros((N+1, 4))
    log_n   = np.zeros((N+1, 3))
    log_d_des = np.zeros((N+1, 3))
    enabled_t = None

    state_cur = state.copy()
    for k in range(N+1):
        t = k*dt
        R_mat = state_cur[6:15].reshape(3,3)
        om_now = state_cur[15:18]
        # Track inertial up vector in body frame (the "n" in eq 15)
        n_body = R_mat.T @ np.array([0,0,1.0])

        log_t[k]   = t
        log_d[k]   = state_cur[0:3]
        log_v[k]   = state_cur[3:6]
        log_om[k]  = om_now
        log_f[k]   = state_cur[18:22]
        log_n[k]   = n_body
        d_des = setpoint(t)
        log_d_des[k] = d_des

        if k == N: break

        if enabled_t is None and np.linalg.norm(om_now) > 10.0:
            enabled_t = t
            print(f" LQR enabled at t = {t:.3f} s")

        if enabled_t is None:
            f_cmd = pre_phase_cmd.copy()
        else:
            f_cmd, _ = ctrl(state_cur, d_des)

        state_cur = rk4_step(dynamics, state_cur, dt, f_cmd)

    print(f" Final position : {state_cur[0:3]}")
    print(f" Final omega    : {state_cur[15:18]}")
    print(f" Final forces   : {state_cur[18:22]}")

    # ----- Plots -----
    plot_equilibrium_curve()
    plot_results(log_t, log_d, log_d_des, log_v, log_om, log_f, log_n, eq, enabled_t)

def plot_equilibrium_curve():
    rhos = np.linspace(0.001, 5.0, 400)
    omn = []; q = []; r = []; nz = []; ny = []; f1l=[]; f2l=[]; Rps=[]
    for rho in rhos:
        try:
            eq = equilibrium_one_prop(rho)
            omn.append(np.linalg.norm(eq['omega_B']))
            q.append(eq['omega_B'][1]); r.append(eq['omega_B'][2])
            nz.append(eq['n_bar'][2]); ny.append(eq['n_bar'][1])
            f1l.append(eq['f'][0]); f2l.append(eq['f'][1])
            Rps.append(eq['R_ps'])
        except Exception:
            for L in (omn,q,r,nz,ny,f1l,f2l,Rps): L.append(np.nan)
    fig, axs = plt.subplots(4,1, figsize=(7,10), sharex=True)
    axs[0].plot(rhos, omn, label='|omega_B|')
    axs[0].plot(rhos, q,   label='q'); axs[0].plot(rhos, r, label='r')
    axs[0].set_ylabel('rad/s'); axs[0].legend(); axs[0].grid(True)
    axs[0].set_title('Fig. 3 replication: equilibrium vs rho = f2/f1')
    axs[1].plot(rhos, ny, label='n_y'); axs[1].plot(rhos, nz, label='n_z')
    axs[1].set_ylabel('primary axis'); axs[1].legend(); axs[1].grid(True)
    axs[2].plot(rhos, f1l, label='f1=f3'); axs[2].plot(rhos, f2l, label='f2')
    axs[2].set_ylabel('Force [N]'); axs[2].legend(); axs[2].grid(True)
    axs[3].plot(rhos, Rps); axs[3].set_ylabel('R_ps [m]'); axs[3].grid(True)
    axs[3].set_xlabel('rho')
    plt.tight_layout()
    plt.savefig('/Users/dikshant/Desktop/uav-project/fig3_equilibrium.png', dpi=130)
    print(" Saved fig3_equilibrium.png")

def plot_results(t, d, d_des, v, om, f, n, eq, enabled_t):
    fig, axs = plt.subplots(5,1, figsize=(9,12), sharex=True)
    axs[0].plot(t, d[:,0], label='x'); axs[0].plot(t, d_des[:,0],'--',label='x_des')
    axs[0].plot(t, d[:,1], label='y'); axs[0].plot(t, d_des[:,1],'--',label='y_des')
    axs[0].set_ylabel('horizontal [m]'); axs[0].legend(); axs[0].grid(True)
    axs[0].set_title('Fig. 4 replication: 1-prop failure flight')
    axs[1].plot(t, d[:,2], label='z'); axs[1].plot(t, d_des[:,2],'--',label='z_des')
    axs[1].set_ylabel('height [m]'); axs[1].legend(); axs[1].grid(True)
    axs[2].plot(t, om[:,0], label='p'); axs[2].plot(t, om[:,1], label='q')
    axs[2].plot(t, om[:,2], label='r')
    for v_, c in zip(eq['omega_B'], ['C0','C1','C2']):
        axs[2].axhline(v_, ls=':', color=c, alpha=0.6)
    axs[2].set_ylabel('omega_B [rad/s]'); axs[2].legend(); axs[2].grid(True)
    axs[3].plot(t, f[:,0], label='f1'); axs[3].plot(t, f[:,1], label='f2')
    axs[3].plot(t, f[:,2], label='f3'); axs[3].plot(t, f[:,3], label='f4')
    axs[3].axhline(F_MAX, ls=':', color='k'); axs[3].axhline(F_MIN, ls=':', color='k')
    axs[3].set_ylabel('Forces [N]'); axs[3].legend(); axs[3].grid(True)
    axs[4].plot(t, n[:,0], label='n_x'); axs[4].plot(t, n[:,1], label='n_y')
    axs[4].plot(t, n[:,2], label='n_z')
    axs[4].set_ylabel('inertial-up in body'); axs[4].legend(); axs[4].grid(True)
    axs[4].set_xlabel('time [s]')
    if enabled_t is not None:
        for ax in axs: ax.axvline(enabled_t, ls='--', color='gray', alpha=0.5)
    plt.tight_layout()
    plt.savefig('/Users/dikshant/Desktop/uav-project/fig4_flight.png', dpi=130)
    print(" Saved fig4_flight.png")

if __name__ == '__main__':
    run()
