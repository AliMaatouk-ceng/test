"""
Stochastic state-space model of a ground vehicle for the IO-VNBD dataset (V- / CAN-bus files).

    x_{k+1} = f(x_k, u_k) + w_k ,   w_k ~ N(0, Q*dt)      (process: dynamics + sensor-bias drift)
    y_k     = h(x_k, u_k) + v_k ,   v_k ~ N(0, R)         (sensors)

Latent state x (11):  px, py, psi, vx, vy, r, b_ax, b_ay, b_r, k_w, delta (road-wheel steer angle)
Inputs u (3):         accelerator [%], brake pressure [psi], overall drive ratio (engine rad/s / wheel rad/s)
Observations y (13):  GPS x/y, GPS speed, GPS course, indicated speed, 4 wheel speeds,
                      yaw rate, longitudinal accel, lateral accel, steering-wheel sensor [deg]

Findings from the real V-S3a.csv (see notes in load_vbox):
  * the steering column is offset by ~+9.4 deg, positive = RIGHT, and CLIPPED at 0 for left turns
    -> it is treated as a censored *measurement* of the latent steer angle, not as an input
  * the 'Gear' columns are unreliable -> drive ratio is derived from rpm / wheel speed

Matrix form (see VehicleSSM.matrices): the model is nonlinear, so constant A, B, C, D only exist after
linearising around an operating point (x*, u*). The matrices returned there are
    x_{k+1} = A x_k + B u_k + c + w_k ,   y_k = C x_k + D u_k + e + v_k
with A = df/dx, B = df/du, C = dh/dx, D = dh/du (all analytic, derived by hand; check_matrices() tests them against
finite differences) and c, e the affine offsets that make the form exact at (x*, u*).

Frame: local ENU, psi measured counter-clockwise from East, body x forward, body y left.
All default physical parameters are rough Ford-Fiesta-class PLACEHOLDERS: fit them (see fit()).
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import numpy as np

G = 9.80665
DEG = np.pi / 180.0
KMH = 1.0 / 3.6

STATE_NAMES = ["px", "py", "psi", "vx", "vy", "r", "b_ax", "b_ay", "b_r", "k_w", "delta"]
INPUT_NAMES = ["pedal_pct", "brake_psi", "gear_ratio"]
OBS_NAMES = ["gps_x", "gps_y", "gps_speed", "gps_course", "ind_speed",
             "w_fl", "w_fr", "w_rl", "w_rr", "yaw_rate", "ax", "ay", "steer_deg"]
IDX_COURSE = OBS_NAMES.index("gps_course")
IDX_PSI = STATE_NAMES.index("psi")

# Column order of the ECU/VBOX files, Table 3 of the Data in Brief paper (29 columns)
VBOX_COLUMNS = [
    "n_sats", "time_s", "lat_deg", "lon_deg", "gps_speed_kmh", "gps_heading_deg",
    "gps_height_km", "gps_vvel_kmh", "dt_s", "steer_deg", "w_fl", "w_fr", "w_rl", "w_rr",
    "yaw_rate_degs", "ind_speed_kmh", "ind_ax_g", "ind_ay_g", "handbrake", "gear_req",
    "gear", "rpm", "coolant_c", "clutch", "brake_psi", "brake_pos", "batt_v",
    "air_temp_c", "pedal_pct",
]


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


@dataclass
class VehicleParams:
    m: float = 1150.0          # mass [kg]
    Iz: float = 1600.0         # yaw inertia [kg m^2]
    a: float = 1.00            # CG -> front axle [m]
    b: float = 1.49            # CG -> rear axle [m]
    track: float = 1.48        # track width [m]
    Cf: float = 65000.0        # front axle cornering stiffness [N/rad]
    Cr: float = 55000.0        # rear axle cornering stiffness [N/rad]
    R0: float = 0.2763         # rolling radius [m]: median(GPS speed / wheel speed) in V-S3a (k_w scales it)
    steer_ratio: float = 16.3  # steering-wheel / road-wheel angle (regression on right turns, R2=0.99)
    steer_offset: float = 9.4  # sensor reading when driving straight [deg]; positive reading = right turn
    k_pedal: float = 0.04      # accel per % pedal at ratio_ref [m/s^2 / %]
    ratio_ref: float = 4.0     # reference overall drive ratio (engine rad/s per wheel rad/s)
    k_brake: float = 0.09      # decel per psi of brake pressure [m/s^2 / psi] (regression on V-S3a)
    c_drag: float = 8.0e-4     # drag / m [1/m]
    c_roll: float = 0.10       # rolling resistance [m/s^2]
    v_min: float = 1.5         # lateral-dynamics speed floor [m/s] (bicycle model is singular at 0)
    n_sub: int = 4             # RK4 sub-steps per sample (lateral dynamics are stiff at low speed)



class VehicleSSM:
    def __init__(self, params: VehicleParams | None = None, dt: float = 0.1,
                 q=None, r=None):
        self.p = params or VehicleParams()
        self.dt = dt
        # continuous-time process noise intensities (per second)
        self.Q = np.diag(q if q is not None else
                         [1e-4, 1e-4, 1e-5, 0.5, 0.1, 0.05, 1e-5, 1e-5, 1e-7, 1e-7, 0.02])
        # measurement std devs -> R
        sd = r if r is not None else [1.0, 1.0, 0.15, 0.08, 0.3, 0.3, 0.3, 0.3, 0.3, 0.05, 0.3, 0.3, 3.0]
        self.R = np.diag(np.square(sd))

    # ---------------------------------------------------------------- dynamics
    def _acc(self, x, u):
        p = self.p
        vx = x[3]
        g = u[2] / p.ratio_ref
        return (g * p.k_pedal * u[0] - p.k_brake * u[1]
                - p.c_drag * vx * abs(vx) - p.c_roll * np.tanh(vx / 0.5))

    def f_cont(self, x, u):
        p = self.p
        _, _, psi, vx, vy, r, *_ = x
        delta = x[10]
        vxs = np.sqrt(vx ** 2 + p.v_min ** 2)           # smooth speed floor
        ax_cmd = self._acc(x, u)
        dvx = ax_cmd + r * vy
        dvy = (-(p.Cf + p.Cr) / (p.m * vxs) * vy
               - (vx + (p.a * p.Cf - p.b * p.Cr) / (p.m * vxs)) * r
               + p.Cf / p.m * delta)
        dr = (-(p.a * p.Cf - p.b * p.Cr) / (p.Iz * vxs) * vy
              - (p.a ** 2 * p.Cf + p.b ** 2 * p.Cr) / (p.Iz * vxs) * r
              + p.a * p.Cf / p.Iz * delta)
        dx = np.zeros_like(x)
        dx[0] = vx * np.cos(psi) - vy * np.sin(psi)
        dx[1] = vx * np.sin(psi) + vy * np.cos(psi)
        dx[2] = r
        dx[3], dx[4], dx[5] = dvx, dvy, dr             # biases, k_w and delta: random walks (dx = 0)
        return dx

    def f(self, x, u, dt=None):
        """One sample step = n_sub RK4 sub-steps (input held constant over the step)."""
        dt = (dt or self.dt) / self.p.n_sub
        for _ in range(self.p.n_sub):
            k1 = self.f_cont(x, u)
            k2 = self.f_cont(x + 0.5 * dt * k1, u)
            k3 = self.f_cont(x + 0.5 * dt * k2, u)
            k4 = self.f_cont(x + dt * k3, u)
            x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return x

    # ------------------------------------------------------------ observation
    def h(self, x, u):
        p = self.p
        px, py, psi, vx, vy, r, b_ax, b_ay, b_r, k_w, delta = x
        dvy = self.f_cont(x, u)[4]
        wheel_r = p.R0 * k_w
        vl, vr = vx - r * p.track / 2, vx + r * p.track / 2
        # front wheels: project hub velocity onto the steered wheel plane
        vfl = vl * np.cos(delta) + (vy + r * p.a) * np.sin(delta)
        vfr = vr * np.cos(delta) + (vy + r * p.a) * np.sin(delta)
        return np.array([
            px, py,
            np.hypot(vx, vy),
            psi + np.arctan2(vy, max(vx, 0.1)),        # course = heading + sideslip
            vx,
            vfl / wheel_r, vfr / wheel_r, vl / wheel_r, vr / wheel_r,
            r + b_r,
            self._acc(x, u) + b_ax,                    # accelerometer sees dvx - r*vy = ax_cmd
            dvy + r * vx + b_ay,
            p.steer_offset - p.steer_ratio * delta / DEG,   # right-positive, clipped at 0 in the real sensor
        ])

    # ------------------------------------------------------ matrix form (A, B, C, D)
    def _ac_bc(self, x, u):
        """Analytic continuous-time Jacobians  Ac = df_c/dx (11x11),  Bc = df_c/du (11x3)."""
        p = self.p
        _, _, psi, vx, vy, r, *_ = x
        pedal, _, ratio = u
        vxs = np.sqrt(vx ** 2 + p.v_min ** 2)
        vxs3 = vxs ** 3
        dax_dvx = -2 * p.c_drag * abs(vx) - p.c_roll * (1 - np.tanh(vx / 0.5) ** 2) / 0.5
        k1 = (p.Cf + p.Cr) / p.m
        k2 = (p.a * p.Cf - p.b * p.Cr) / p.m
        j2 = (p.a * p.Cf - p.b * p.Cr) / p.Iz
        j3 = (p.a ** 2 * p.Cf + p.b ** 2 * p.Cr) / p.Iz
        Ac = np.zeros((len(STATE_NAMES), len(STATE_NAMES)))
        Ac[0, 2] = -vx * np.sin(psi) - vy * np.cos(psi)
        Ac[0, 3], Ac[0, 4] = np.cos(psi), -np.sin(psi)
        Ac[1, 2] = vx * np.cos(psi) - vy * np.sin(psi)
        Ac[1, 3], Ac[1, 4] = np.sin(psi), np.cos(psi)
        Ac[2, 5] = 1.0
        Ac[3, 3], Ac[3, 4], Ac[3, 5] = dax_dvx, r, vy
        Ac[4, 3] = k1 * vy * vx / vxs3 - r * (1 - k2 * vx / vxs3)
        Ac[4, 4] = -k1 / vxs
        Ac[4, 5] = -(vx + k2 / vxs)
        Ac[4, 10] = p.Cf / p.m
        Ac[5, 3] = j2 * vy * vx / vxs3 + j3 * r * vx / vxs3
        Ac[5, 4] = -j2 / vxs
        Ac[5, 5] = -j3 / vxs
        Ac[5, 10] = p.a * p.Cf / p.Iz
        Bc = np.zeros((len(STATE_NAMES), len(INPUT_NAMES)))
        Bc[3] = [ratio / p.ratio_ref * p.k_pedal, -p.k_brake, p.k_pedal * pedal / p.ratio_ref]
        return Ac, Bc

    def _discrete_jac(self, x, u):
        """Exact Jacobians (A, B) of the implemented discrete map f(x,u) = n_sub x RK4 steps.
        Chain rule through the RK4 stages with the analytic Ac, Bc (no finite differences)."""
        n = len(x)
        h = self.dt / self.p.n_sub
        I = np.eye(n)
        A, B = I.copy(), np.zeros((n, len(u)))
        for _ in range(self.p.n_sub):
            k1 = self.f_cont(x, u)
            x2 = x + 0.5 * h * k1
            k2 = self.f_cont(x2, u)
            x3 = x + 0.5 * h * k2
            k3 = self.f_cont(x3, u)
            x4 = x + h * k3
            (A1, Bc), (A2, _), (A3, _), (A4, _) = (self._ac_bc(x, u), self._ac_bc(x2, u),
                                                  self._ac_bc(x3, u), self._ac_bc(x4, u))
            K1 = A1
            K2 = A2 @ (I + 0.5 * h * K1)
            K3 = A3 @ (I + 0.5 * h * K2)
            K4 = A4 @ (I + h * K3)
            L1 = Bc
            L2 = Bc + A2 @ (0.5 * h * L1)
            L3 = Bc + A3 @ (0.5 * h * L2)
            L4 = Bc + A4 @ (h * L3)
            As = I + h / 6 * (K1 + 2 * K2 + 2 * K3 + K4)
            Bs = h / 6 * (L1 + 2 * L2 + 2 * L3 + L4)
            A, B = As @ A, As @ B + Bs
            x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + self.f_cont(x4, u))
        return A, B

    def matrices(self, x, u, discrete=True):
        """Linearise the model at (x, u) and return a dict with
             continuous:  Ac (11x11), Bc (11x3)      xdot = Ac x + Bc u + ...
             discrete:    A  (11x11), B  (11x3)      x_{k+1} = A x_k + B u_k + c
             output:      C  (13x11), D  (13x3)      y_k     = C x_k + D u_k + e
             offsets:     c, e  (exact at the operating point)
        All analytic; this is the single source of truth used by the EKF."""
        p = self.p
        px, py, psi, vx, vy, r, b_ax, b_ay, b_r, k_w, de = x
        pedal, brake, ratio = u
        n, nu, ny = len(STATE_NAMES), len(INPUT_NAMES), len(OBS_NAMES)
        Ac, Bc = self._ac_bc(x, u)

        # ---- output matrices  C = dh/dx ,  D = dh/du
        C, D = np.zeros((ny, n)), np.zeros((ny, nu))
        C[0, 0] = C[1, 1] = 1.0
        sp = np.hypot(vx, vy)
        if sp > 1e-6:
            C[2, 3], C[2, 4] = vx / sp, vy / sp
        vxm = max(vx, 0.1)
        C[3, 2] = 1.0
        C[3, 4] = vxm / (vxm ** 2 + vy ** 2)
        if vx > 0.1:
            C[3, 3] = -vy / (vxm ** 2 + vy ** 2)
        C[4, 3] = 1.0
        wr = p.R0 * k_w
        vl, vr = vx - r * p.track / 2, vx + r * p.track / 2
        cd, sd = np.cos(de), np.sin(de)
        vfl, vfr = vl * cd + (vy + r * p.a) * sd, vr * cd + (vy + r * p.a) * sd
        for row, vh, vside, dvr in [(5, vfl, vl, -p.track / 2), (6, vfr, vr, p.track / 2)]:
            C[row, 3], C[row, 4] = cd / wr, sd / wr
            C[row, 5] = (dvr * cd + p.a * sd) / wr
            C[row, 9] = -vh / wr / k_w
            C[row, 10] = (-vside * sd + (vy + r * p.a) * cd) / wr
        C[7, 3], C[7, 5], C[7, 9] = 1 / wr, -p.track / 2 / wr, -vl / wr / k_w
        C[8, 3], C[8, 5], C[8, 9] = 1 / wr, p.track / 2 / wr, -vr / wr / k_w
        C[9, 5] = C[9, 8] = 1.0
        C[10, 3], C[10, 6] = Ac[3, 3], 1.0
        D[10] = Bc[3]                                    # accelerometer feed-through from the inputs
        C[11, 3] = Ac[4, 3] + r
        C[11, 4], C[11, 5] = Ac[4, 4], Ac[4, 5] + vx
        C[11, 10], C[11, 7] = Ac[4, 10], 1.0
        C[12, 10] = -p.steer_ratio / DEG
        out = dict(Ac=Ac, Bc=Bc, C=C, D=D, e=self.h(x, u) - C @ x - D @ u)

        if discrete:
            out["A"], out["B"] = self._discrete_jac(x, u)
            out["c"] = self.f(x, u) - out["A"] @ x - out["B"] @ u
        return out

    # ------------------------------------------------------------- simulation
    def default_x0(self, psi0=0.0, v0=0.0):
        return np.array([0, 0, psi0, v0, 0, 0, 0, 0, 0, 1.0, 0.0])

    def simulate(self, x0, U, rng=None, noise=True):
        """Draw a trajectory from the stochastic model. U: (T, 4). Returns X (T,10), Y (T,12)."""
        rng = rng or np.random.default_rng(0)
        T = len(U)
        X, Y = np.zeros((T, len(x0))), np.zeros((T, len(OBS_NAMES)))
        x = x0.copy()
        L = np.sqrt(np.diag(self.Q) * self.dt)
        for k in range(T):
            X[k] = x
            Y[k] = self.h(x, U[k]) + (rng.normal(size=len(OBS_NAMES)) * np.sqrt(np.diag(self.R)) if noise else 0)
            x = self.f(x, U[k]) + (rng.normal(size=len(x)) * L if noise else 0)
        return X, Y

    # -------------------------------------------------------------------- EKF
    @staticmethod
    def _jac(fun, x, eps=1e-6):
        y0 = fun(x)
        J = np.zeros((len(y0), len(x)))
        for i in range(len(x)):
            d = np.zeros_like(x)
            d[i] = eps * max(1.0, abs(x[i]))
            J[:, i] = (fun(x + d) - fun(x - d)) / (2 * d[i])
        return J

    def ekf(self, U, Y, x0, P0, analytic=True):
        """Extended Kalman filter built on matrices() (analytic A, C); analytic=False uses finite differences. NaN entries in Y are treated as missing (GPS outages, etc.).
        Returns filtered states, covariances and the negative log-likelihood (for parameter fitting)."""
        T, n = len(Y), len(x0)
        Xs, Ps = np.zeros((T, n)), np.zeros((T, n, n))
        x, P, nll = x0.copy(), P0.copy(), 0.0
        Qd = self.Q * self.dt
        for k in range(T):
            m = ~np.isnan(Y[k])
            if m.any():
                yhat = self.h(x, U[k])
                H = (self.matrices(x, U[k], discrete=False)["C"] if analytic
                     else self._jac(lambda z: self.h(z, U[k]), x))[m]
                e = (Y[k] - yhat)
                e[IDX_COURSE] = wrap(e[IDX_COURSE]) if m[IDX_COURSE] else 0.0
                e = e[m]
                S = H @ P @ H.T + self.R[np.ix_(m, m)]
                K = np.linalg.solve(S, H @ P).T
                x = x + K @ e
                I_KH = np.eye(n) - K @ H
                P = I_KH @ P @ I_KH.T + K @ self.R[np.ix_(m, m)] @ K.T      # Joseph form
                nll += 0.5 * (e @ np.linalg.solve(S, e) + np.linalg.slogdet(S)[1])
            Xs[k], Ps[k] = x, P
            if k < T - 1:
                A = (self.matrices(x, U[k])["A"] if analytic
                     else self._jac(lambda z: self.f(z, U[k]), x))
                x = self.f(x, U[k])
                P = A @ P @ A.T + Qd
        return Xs, Ps, nll

    # ----------------------------------------------------------------- fitting
    def fit(self, U, Y, x0, P0, names=("Cf", "Cr", "k_pedal", "k_brake", "R0"), maxiter=40):
        """Maximum-likelihood fit of selected physical parameters (log-parametrised, via EKF likelihood)."""
        from scipy.optimize import minimize
        theta0 = np.log([getattr(self.p, n) for n in names])

        def obj(theta):
            for n, v in zip(names, np.exp(theta)):
                setattr(self.p, n, float(v))
            try:
                return self.ekf(U, Y, x0, P0)[2]
            except np.linalg.LinAlgError:
                return 1e12

        res = minimize(obj, theta0, method="Nelder-Mead", options={"maxiter": maxiter})
        for n, v in zip(names, np.exp(res.x)):
            setattr(self.p, n, float(v))
        return res


def check_matrices(model=None, trials=100, seed=0):
    """Compare analytic A, B, C, D (and Ac, Bc) with central finite differences at random operating points."""
    m = model or VehicleSSM()
    rng = np.random.default_rng(seed)
    worst = dict.fromkeys(["A", "B", "C", "D", "Ac", "Bc"], 0.0)
    for _ in range(trials):
        x = np.array([0, 0, rng.uniform(-3, 3), rng.uniform(0.5, 30), rng.uniform(-1, 1), rng.uniform(-.5, .5),
                      0.05, -0.05, 0.01, rng.uniform(.95, 1.05), rng.uniform(-.1, .1)])
        u = np.array([rng.uniform(0, 40), rng.uniform(0, 30), rng.uniform(3, 9)])
        M = m.matrices(x, u)
        ref = {"A": m._jac(lambda z: m.f(z, u), x), "B": m._jac(lambda z: m.f(x, z), u),
               "C": m._jac(lambda z: m.h(z, u), x), "D": m._jac(lambda z: m.h(x, z), u),
               "Ac": m._jac(lambda z: m.f_cont(z, u), x), "Bc": m._jac(lambda z: m.f_cont(x, z), u)}
        for k in worst:
            worst[k] = max(worst[k], np.max(np.abs(M[k] - ref[k])) / max(1, np.max(np.abs(ref[k]))))
    return worst


# ------------------------------------------------------------------ data I/O
def load_vbox(path, p: VehicleParams | None = None, use_course=True):
    """Load a V-* CSV from IO-VNBD -> (t, U, Y, x0). Columns are assigned by position (Table 3 of the
    paper; verified against V-S3a.csv, whose header has leading spaces and the same order).

    Data quirks handled here (found on V-S3a.csv):
      * steering column: offset ~9.4 deg, positive = right, reads exactly 0 for left turns
        (98.8% of left turns >3 deg/s) -> zeros become NaN (censored, no information)
      * 'Gear' is ~constant (3) and 'Gear Requested' is not the engaged gear -> the overall drive ratio
        is derived as engine speed / mean rear wheel speed (forward-filled when stationary)
      * 'No of GPS satellites' holds values up to 138 (not a count) -> NOT used for outage masking;
        mask GPS yourself (e.g. NaN rows, or the paper's 'GPS outages' index file)
      * brake pressure has small negative offsets -> clipped at 0
      * accelerator column header says '(0 or 1)' but values are 0-62 % -> kept as %
    """
    import pandas as pd
    df = pd.read_csv(path)
    df = df.iloc[:, :29]
    df.columns = VBOX_COLUMNS
    p = p or VehicleParams()

    lat, lon = np.radians(df.lat_deg.values), np.radians(df.lon_deg.values)
    Rearth = 6378137.0
    px = Rearth * np.cos(lat[0]) * (lon - lon[0])          # East
    py = Rearth * (lat - lat[0])                            # North
    spd = df.gps_speed_kmh.values * KMH
    course = wrap(np.pi / 2 - np.radians(df.gps_heading_deg.values))   # compass (cw from N) -> ENU (ccw from E)
    course = np.where((spd > 1.0) & use_course, course, np.nan)

    w_rear = 0.5 * (df.w_rl.values + df.w_rr.values)
    ratio = np.where(w_rear > 3.0, df.rpm.values * 2 * np.pi / 60 / np.maximum(w_rear, 1e-6), np.nan)
    ratio = pd.Series(np.clip(ratio, 2.4, 10.0)).ffill().bfill().values

    steer = df.steer_deg.values.astype(float)
    steer[steer <= 0.0] = np.nan                            # censored (left turns)

    U = np.column_stack([df.pedal_pct.values, np.clip(df.brake_psi.values, 0, None), ratio])
    Y = np.column_stack([
        px, py, spd, course, df.ind_speed_kmh.values * KMH,
        df.w_fl.values, df.w_fr.values, df.w_rl.values, df.w_rr.values,
        df.yaw_rate_degs.values * DEG, df.ind_ax_g.values * G, df.ind_ay_g.values * G, steer])
    x0 = np.array([0, 0, 0 if np.isnan(course[0]) else course[0], spd[0], 0, 0, 0, 0, 0, 1.0, 0.0])
    return df.time_s.values, U, Y, x0


if __name__ == "__main__":
    # Self-test on synthetic data: simulate -> filter -> compare with the hidden truth
    model = VehicleSSM()
    T = 1200
    t = np.arange(T) * model.dt
    steer = 0.04 * np.sin(2 * np.pi * t / 25) * (t > 10)
    pedal = np.where(t < 15, 35.0, 12.0 + 4 * np.sin(2 * np.pi * t / 40))
    brake = np.where((t > 80) & (t < 85), 15.0, 0.0)
    ratio = np.where(t < 4, 9.0, np.where(t < 9, 6.0, 4.0))
    U = np.column_stack([pedal, brake, ratio])

    # truth: steering is a latent state -> drive it by overriding delta each step
    x0_true = model.default_x0(psi0=0.5, v0=2.0)
    rng = np.random.default_rng(1)
    X = np.zeros((T, len(x0_true))); Y = np.zeros((T, len(OBS_NAMES)))
    x = x0_true.copy()
    for k in range(T):
        x[10] = steer[k]
        X[k] = x
        Y[k] = model.h(x, U[k]) + rng.normal(size=len(OBS_NAMES)) * np.sqrt(np.diag(model.R))
        x = model.f(x, U[k]) + rng.normal(size=len(x)) * np.sqrt(np.diag(model.Q) * model.dt)
    Y[300:450, [0, 1, 2, 3]] = np.nan                       # GPS outage
    Y[:, -1] = np.where(Y[:, -1] <= 0, np.nan, Y[:, -1])     # steering sensor clipped at 0 (left turns)

    x0 = x0_true + np.array([3, -3, 0.2, 0.5, 0, 0, 0, 0, 0, 0.02, 0])
    P0 = np.diag([10, 10, 0.2, 1, 0.5, 0.1, 0.05, 0.05, 0.02, 0.05, 0.1]) ** 2
    Xf, Pf, nll = model.ekf(U, Y, x0, P0)
    rmse = np.sqrt(np.mean((Xf - X) ** 2, axis=0))
    print("EKF NLL:", round(nll, 1))
    for n, e in zip(STATE_NAMES, rmse):
        print(f"  RMSE {n:5s}: {e:.4f}")
