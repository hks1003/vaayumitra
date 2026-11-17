"""
VaayuMitra physics core: fishing boat + sail force balance -> fuel per hour.

Design choices (documented for the poster's methods panel):
- Boat baseline is CALIBRATED to measured Indian fuel rates, not a generic regression:
  15 m single-day trawler (90-140 hp, Kerala/TN) steaming 25.5 L/h, trawling 20.4 L/h
  [CIFT LNG-diesel dual-fuel trials off Cochin, Fishery Technology; CMFRI 2014 trawl-ban
  report for 90-140 hp @ 15 m LOA]. Ship speed is held fixed (fishers keep their schedule);
  the sail only unloads the engine.
- Sail aerodynamics: lifting-line wing with finite aspect ratio, CL capped at CLmax, and a
  flat-plate post-stall branch; the trim angle is optimised every time step. Coefficients are
  generic (soft sail vs rigid wing) and are varied +-20 % in the sensitivity study.
- Side-force penalty: hull treated as a low-aspect wing (span = 2*draft, mirror image);
  induced drag D_i = Fy^2 / (0.5 rho_w V^2 pi b^2 e_h).
- Safety: sail is reefed so the steady heel stays <= HEEL_MAX, and furled when Hs > HS_MAX
  or true wind > TWS_MAX.
- Fuel: Willans-line engine model (idle fuel + linear in brake power).
"""
from dataclasses import dataclass, field

import numpy as np

RHO_AIR = 1.225      # kg/m3
RHO_W = 1025.0       # kg/m3
G = 9.81
KN = 0.514444        # m/s per knot
DIESEL_KG_L = 0.84   # kg per litre
CO2_KG_PER_L = 2.68  # kg CO2 per litre diesel (IPCC tank-to-wake)


@dataclass
class Boat:
    name: str = "15 m single-day trawler"
    L: float = 15.0            # m
    B: float = 5.0             # m
    T: float = 1.8             # m draft
    disp_t: float = 55.0       # displacement, tonnes
    GM: float = 0.7            # m metacentric height
    P_rated_kW: float = 90.0   # ~120 hp
    eta_D: float = 0.50        # propulsive efficiency (fixed-pitch, small boat)
    e_hull: float = 0.6        # hull lateral-lift efficiency
    # calibration targets (measured)
    V_steam_kn: float = 8.0
    fuel_steam_Lph: float = 25.5
    V_trawl_kn: float = 3.0
    fuel_trawl_Lph: float = 20.4
    idle_frac: float = 0.08    # idle fuel as fraction of rated fuel (Willans intercept)
    sfc_gkWh: float = 230.0    # marginal specific fuel consumption
    op_mode: str = "trawl"     # fishing operation: 'trawl' (towing) or 'drift' (gillnet, engine ~off)
    sail_set: str = "big"      # which sail sizes fit this boat
    V_run_kn: float = 0.0      # operating steaming speed (0 -> calibration speed); slow-steaming lever

    def __post_init__(self):
        # Willans line: fuel_kg_h = m0 + k*P,  k from marginal SFC
        self.k = self.sfc_gkWh / 1000.0
        m0_raw = self.idle_frac * self.P_rated_kW * self.k
        # Brake power that reproduces measured fuel at each mode (calm water)
        self.P_steam = (self.fuel_steam_Lph * DIESEL_KG_L - m0_raw) / self.k
        self.P_trawl = (self.fuel_trawl_Lph * DIESEL_KG_L - m0_raw) / self.k
        self.m0 = m0_raw
        # Effective resistance / tow force implied by the calibration (N)
        self.R_steam = self.eta_D * self.P_steam * 1e3 / (self.V_steam_kn * KN)
        self.R_trawl = self.eta_D * self.P_trawl * 1e3 / (self.V_trawl_kn * KN)
        # calm-water resistance scaling R ~ V^2.3 around the steaming point
        self.n_R = 2.3
        if self.V_run_kn <= 0:
            self.V_run_kn = self.V_steam_kn

    def fuel_Lph(self, P_kW):
        P = np.clip(P_kW, 0.0, self.P_rated_kW * 1.5)  # allow overload so savings are not masked
        return (self.m0 + self.k * P) / DIESEL_KG_L

    def added_wave_resistance(self, Hs, V_ms):
        # Simple Hs^2 model (order of a few % at Hs~1.5 m); identical for sail / no-sail
        return 0.05 * RHO_W * G * Hs**2 * self.B**2 / self.L * np.clip(V_ms / (self.V_steam_kn * KN), 0.3, 1.0)


@dataclass
class Sail:
    name: str
    area: float      # m2
    AR: float        # aspect ratio
    CLmax: float
    CD0: float
    e: float = 0.8   # span efficiency
    h_ce: float = 5.0  # height of centre of effort above waterline, m
    cost_inr: float = 0.0


SAIL_SETS = {
    "big": {  # for the 15 m trawler
        # low-cost soft sail on existing mast (canvas/polyester, local rigging)
        "soft": Sail("Soft lug-type sail", area=40.0, AR=1.6, CLmax=1.3, CD0=0.10, e=0.7, h_ce=5.0),
        # small rigid/semi-rigid wing sail (higher L/D, higher cost)
        "wing": Sail("Rigid wing sail", area=25.0, AR=2.5, CLmax=1.15, CD0=0.03, e=0.85, h_ce=5.5),
    },
    "small": {  # for the 8.5 m FRP boat; 18 m2 dipping lug as in the FAO/BOBP Chennai trials
        "soft": Sail("Soft lug-type sail", area=18.0, AR=1.3, CLmax=1.3, CD0=0.10, e=0.7, h_ce=2.8),
        "wing": Sail("Rigid wing sail", area=10.0, AR=2.5, CLmax=1.15, CD0=0.03, e=0.85, h_ce=3.0),
    },
}
SAILS = SAIL_SETS["big"]  # backwards-compatible default

BOATS = {
    # 15 m single-day trawler, Kerala/TN; fuel from CIFT Cochin trials (25.5 / 20.4 L/h)
    "trawler": dict(),
    # 8.5 m FRP beach-landing / gillnet boat; FAO small-vessel manual: diesel ~3 L/h at 8 kn
    "frp": dict(name="8.5 m FRP gillnet boat", L=8.5, B=2.0, T=0.6, disp_t=2.5, GM=0.45,
                P_rated_kW=12.0, eta_D=0.45, e_hull=0.5, V_steam_kn=8.0, fuel_steam_Lph=3.0,
                V_trawl_kn=0.5, fuel_trawl_Lph=0.4, op_mode="drift", sail_set="small"),
}

HEEL_MAX_DEG = 8.0   # comfort/safety limit for a working fishing boat
HS_MAX = 2.5         # furl above this significant wave height (m)
TWS_MAX = 14.0       # furl above this true wind speed (m/s)

_ALPHA = np.radians(np.linspace(0.0, 90.0, 91))  # trim-angle search grid


def sail_coeffs(s: Sail, alpha=_ALPHA):
    """CL, CD vs angle of attack: attached lifting line, capped; flat-plate post-stall envelope."""
    a2d = 2 * np.pi
    a3d = a2d / (1 + a2d / (np.pi * s.AR * s.e))
    cl_att = np.minimum(a3d * alpha, s.CLmax)
    cd_att = s.CD0 + cl_att**2 / (np.pi * s.AR * s.e)
    cl_ps = 1.1 * np.sin(2 * alpha)                  # flat plate, finite AR
    cd_ps = s.CD0 + 1.2 * np.sin(alpha) ** 2
    alpha_stall = s.CLmax / a3d
    post = alpha > alpha_stall
    cl = np.where(post, np.minimum(cl_ps, s.CLmax), cl_att)
    cd = np.where(post, np.maximum(cd_ps, cd_att), cd_att)
    return cl, cd


def apparent_wind(tws, twd_from_rad, heading_rad, V_ms):
    """Apparent wind speed and angle off the bow (0 = head wind, pi = dead astern)."""
    # true wind velocity vector (blowing TO) in east/north
    wx, wy = -tws * np.sin(twd_from_rad), -tws * np.cos(twd_from_rad)
    bx, by = V_ms * np.sin(heading_rad), V_ms * np.cos(heading_rad)
    ax, ay = wx - bx, wy - by                       # air relative to boat
    aws = np.hypot(ax, ay)
    # direction air comes FROM relative to bow
    from_ang = np.arctan2(-ax, -ay) - heading_rad
    awa = np.abs((from_ang + np.pi) % (2 * np.pi) - np.pi)
    return aws, awa


def sail_net_saving(boat: Boat, s: Sail, aws, awa, V_ms, Hs, tws):
    """
    Vectorised over arrays of conditions. Returns (thrust_N, side_N, reef_frac, net_thrust_N).
    Trim angle and reef fraction are chosen to maximise net thrust (thrust - induced hull drag)
    subject to the heel limit.
    """
    aws, awa, V_ms, Hs, tws = map(np.atleast_1d, (aws, awa, V_ms, Hs, tws))
    cl, cd = sail_coeffs(s)                           # (A,)
    b = awa[:, None]                                  # (N,1)
    # forward (thrust) and lateral coefficients for each trim angle
    ct = cl * np.sin(b) - cd * np.cos(b)              # (N,A)
    cs = np.abs(cl * np.cos(b) + cd * np.sin(b))
    q = 0.5 * RHO_AIR * aws[:, None] ** 2 * s.area

    # heel limit -> max allowable side force; reef fraction r in (0,1]
    disp_N = boat.disp_t * 1000 * G
    Fy_max = disp_N * boat.GM * np.sin(np.radians(HEEL_MAX_DEG)) / s.h_ce
    best = np.full(aws.shape, -np.inf)
    out = np.zeros((4, aws.size))
    span = 2 * boat.T
    q_w = 0.5 * RHO_W * np.maximum(V_ms, 0.5) ** 2 * np.pi * span**2 * boat.e_hull
    for r in (1.0, 0.75, 0.5, 0.25):
        Fx = r * q * ct
        Fy = r * q * cs
        ok = Fy <= Fy_max
        D_i = Fy**2 / q_w[:, None]
        net = np.where(ok, Fx - D_i, -np.inf)
        j = np.argmax(net, axis=1)
        nbest = net[np.arange(net.shape[0]), j]
        better = nbest > best
        best = np.where(better, nbest, best)
        out[0] = np.where(better, Fx[np.arange(Fx.shape[0]), j], out[0])
        out[1] = np.where(better, Fy[np.arange(Fy.shape[0]), j], out[1])
        out[2] = np.where(better, r, out[2])
    # furl when unsafe or when sail does not help
    furl = (Hs > HS_MAX) | (tws > TWS_MAX) | (best <= 0) | ~np.isfinite(best)
    out[3] = np.where(furl, 0.0, best)
    out[0][furl] = 0.0
    out[1][furl] = 0.0
    out[2][furl] = 0.0
    return out[0], out[1], out[2], out[3]


def fuel_rate(boat: Boat, mode, V_ms, Hs, net_sail_N=0.0):
    """Fuel L/h for a mode ('steam' or 'trawl') given sea state and net sail thrust."""
    V_ms, Hs, net_sail_N = (np.atleast_1d(np.asarray(x, dtype=float)) for x in (V_ms, Hs, net_sail_N))
    if mode == "drift":  # gillnet drifting/hauling: engine near idle, sail not used
        return np.full_like(V_ms, boat.fuel_trawl_Lph, dtype=float)
    if mode == "steam":
        R = boat.R_steam * (V_ms / (boat.V_steam_kn * KN)) ** boat.n_R
    else:  # trawl: tow force dominates, hull resistance small at 3 kn
        R = np.full_like(np.atleast_1d(V_ms), boat.R_trawl, dtype=float)
    R = R + boat.added_wave_resistance(Hs, V_ms)
    R_eng = np.maximum(R - net_sail_N, 0.15 * R)   # engine keeps >=15 % (steerage, gear)
    P_kW = R_eng * V_ms / boat.eta_D / 1e3
    return boat.fuel_Lph(P_kW)


if __name__ == "__main__":
    # Self-check: calibration reproduces measured fuel in calm water; sail polar sanity
    bt = Boat()
    Vs, Vt = bt.V_steam_kn * KN, bt.V_trawl_kn * KN
    print(f"P_steam {bt.P_steam:.1f} kW  P_trawl {bt.P_trawl:.1f} kW  (rated {bt.P_rated_kW})")
    print(f"R_steam {bt.R_steam/1e3:.2f} kN  tow {bt.R_trawl/1e3:.2f} kN")
    print("calm steam L/h", fuel_rate(bt, "steam", Vs, 0.0)[0], "target", bt.fuel_steam_Lph)
    print("calm trawl L/h", fuel_rate(bt, "trawl", Vt, 0.0)[0], "target", bt.fuel_trawl_Lph)
    for key, s in SAILS.items():
        print(f"\n{s.name}: steaming 8 kn, TWS 7 m/s, Hs 1 m")
        for twa in (30, 60, 90, 120, 150, 180):
            aws, awa = map(np.atleast_1d, apparent_wind(7.0, np.radians(twa), 0.0, Vs))
            Fx, Fy, r, net = sail_net_saving(bt, s, aws, awa, np.array([Vs]), np.array([1.0]), np.array([7.0]))
            f0 = fuel_rate(bt, "steam", Vs, 1.0)[0]
            f1 = fuel_rate(bt, "steam", Vs, 1.0, net)[0]
            print(f"  TWA {twa:3d}: AWA {np.degrees(awa[0]):5.1f}  thrust {Fx[0]:6.0f} N  reef {r[0]:.2f}"
                  f"  fuel {f0:5.1f}->{f1:5.1f} L/h  saving {100*(1-f1/f0):4.1f}%")
