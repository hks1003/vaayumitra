/*
 * VaayuMitra physics core - JavaScript port of src/physics.py + the route builder in src/simulate.py
 * and evaluate() in app/app.py, so the advisor runs entirely in the browser (GitHub Pages).
 * Checked against the Python version by tests/check_js_port.py (identical results to < 0.01 L).
 */
(function (root) {
  "use strict";
  const RHO_AIR = 1.225, RHO_W = 1025.0, G = 9.81, KN = 0.514444, DIESEL_KG_L = 0.84;
  const HEEL_MAX_DEG = 8.0, HS_MAX = 2.5, TWS_MAX = 14.0;
  const D2R = Math.PI / 180;

  const BOAT_DEFAULT = {
    name: "15 m single-day trawler", L: 15.0, B: 5.0, T: 1.8, disp_t: 55.0, GM: 0.7, P_rated_kW: 90.0,
    eta_D: 0.5, e_hull: 0.6, V_steam_kn: 8.0, fuel_steam_Lph: 25.5, V_trawl_kn: 3.0, fuel_trawl_Lph: 20.4,
    idle_frac: 0.08, sfc_gkWh: 230.0, op_mode: "trawl", sail_set: "big", V_run_kn: 0.0,
  };
  const BOATS = {
    trawler: {},
    frp: { name: "8.5 m FRP gillnet boat", L: 8.5, B: 2.0, T: 0.6, disp_t: 2.5, GM: 0.45, P_rated_kW: 12.0,
      eta_D: 0.45, e_hull: 0.5, V_steam_kn: 8.0, fuel_steam_Lph: 3.0, V_trawl_kn: 0.5, fuel_trawl_Lph: 0.4,
      op_mode: "drift", sail_set: "small" },
  };
  const SAIL_SETS = {
    big: {
      soft: { name: "Soft lug-type sail", area: 40.0, AR: 1.6, CLmax: 1.3, CD0: 0.10, e: 0.7, h_ce: 5.0 },
      wing: { name: "Rigid wing sail", area: 25.0, AR: 2.5, CLmax: 1.15, CD0: 0.03, e: 0.85, h_ce: 5.5 },
    },
    small: {
      soft: { name: "Soft lug-type sail", area: 18.0, AR: 1.3, CLmax: 1.3, CD0: 0.10, e: 0.7, h_ce: 2.8 },
      wing: { name: "Rigid wing sail", area: 10.0, AR: 2.5, CLmax: 1.15, CD0: 0.03, e: 0.85, h_ce: 3.0 },
    },
  };
  const HARBOURS = {
    Chennai: [13.12, 80.30, [60, 90, 120]],
    Kochi: [10.18, 76.17, [225, 255, 285]],
    Visakhapatnam: [17.69, 83.30, [100, 130, 160]],
  };
  const TRAWL_H = 6;

  function makeBoat(key, V_run_kn) {
    const b = Object.assign({}, BOAT_DEFAULT, BOATS[key], { V_run_kn: V_run_kn || 0 });
    b.k = b.sfc_gkWh / 1000.0;
    b.m0 = b.idle_frac * b.P_rated_kW * b.k;
    b.P_steam = (b.fuel_steam_Lph * DIESEL_KG_L - b.m0) / b.k;
    b.P_trawl = (b.fuel_trawl_Lph * DIESEL_KG_L - b.m0) / b.k;
    b.R_steam = b.eta_D * b.P_steam * 1e3 / (b.V_steam_kn * KN);
    b.R_trawl = b.eta_D * b.P_trawl * 1e3 / (b.V_trawl_kn * KN);
    b.n_R = 2.3;
    if (b.V_run_kn <= 0) b.V_run_kn = b.V_steam_kn;
    return b;
  }

  const clip = (x, lo, hi) => Math.min(Math.max(x, lo), hi);
  const fuelLph = (b, P_kW) => (b.m0 + b.k * clip(P_kW, 0, b.P_rated_kW * 1.5)) / DIESEL_KG_L;
  const addedWave = (b, Hs, V) =>
    0.05 * RHO_W * G * Hs * Hs * b.B * b.B / b.L * clip(V / (b.V_steam_kn * KN), 0.3, 1.0);

  const ALPHA = Array.from({ length: 91 }, (_, i) => i * D2R);   // 0..90 deg, 1 deg steps
  function sailCoeffs(s) {
    const a2d = 2 * Math.PI, a3d = a2d / (1 + a2d / (Math.PI * s.AR * s.e)), aStall = s.CLmax / a3d;
    return ALPHA.map((a) => {
      const clAtt = Math.min(a3d * a, s.CLmax), cdAtt = s.CD0 + clAtt * clAtt / (Math.PI * s.AR * s.e);
      if (a > aStall) {
        const clPs = 1.1 * Math.sin(2 * a), cdPs = s.CD0 + 1.2 * Math.sin(a) ** 2;
        return [Math.min(clPs, s.CLmax), Math.max(cdPs, cdAtt)];
      }
      return [clAtt, cdAtt];
    });
  }

  function apparentWind(tws, twdFrom, hdg, V) {
    const wx = -tws * Math.sin(twdFrom), wy = -tws * Math.cos(twdFrom);
    const ax = wx - V * Math.sin(hdg), ay = wy - V * Math.cos(hdg);
    const aws = Math.hypot(ax, ay);
    const fromAng = Math.atan2(-ax, -ay) - hdg;
    const m = ((fromAng + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI);   // python-style modulo
    return [aws, Math.abs(m - Math.PI)];
  }

  /** net useful sail thrust (N) after trim/reef optimisation, heel limit, hull induced drag, furling */
  function sailNet(b, s, coeffs, aws, awa, V, Hs, tws) {
    const q = 0.5 * RHO_AIR * aws * aws * s.area;
    const FyMax = b.disp_t * 1000 * G * b.GM * Math.sin(HEEL_MAX_DEG * D2R) / s.h_ce;
    const span = 2 * b.T;
    const qw = 0.5 * RHO_W * Math.max(V, 0.5) ** 2 * Math.PI * span * span * b.e_hull;
    let best = -Infinity;
    for (const r of [1.0, 0.75, 0.5, 0.25]) {
      let nb = -Infinity;
      for (const [cl, cd] of coeffs) {
        const ct = cl * Math.sin(awa) - cd * Math.cos(awa);
        const cs = Math.abs(cl * Math.cos(awa) + cd * Math.sin(awa));
        const Fx = r * q * ct, Fy = r * q * cs;
        const net = Fy <= FyMax ? Fx - Fy * Fy / qw : -Infinity;
        if (net > nb) nb = net;
      }
      if (nb > best) best = nb;
    }
    const furl = Hs > HS_MAX || tws > TWS_MAX || !(best > 0) || !isFinite(best);
    return furl ? 0 : best;
  }

  function fuelRate(b, mode, V, Hs, net) {
    if (mode === "drift") return b.fuel_trawl_Lph;
    let R = mode === "steam" ? b.R_steam * (V / (b.V_steam_kn * KN)) ** b.n_R : b.R_trawl;
    R += addedWave(b, Hs, V);
    const Reng = Math.max(R - (net || 0), 0.15 * R);
    return fuelLph(b, Reng * V / b.eta_D / 1e3);
  }

  function destination(lat, lon, brg, nm) {
    const d = nm * 1852.0, br = brg * D2R;
    return [lat + d * Math.cos(br) / 111320.0, lon + d * Math.sin(br) / (111320.0 * Math.cos(lat * D2R))];
  }

  function buildRoute(b, harbour, bearing, dist) {
    const [la, lo] = HARBOURS[harbour], steps = [];
    const tLeg = dist / b.V_run_kn, n = Math.ceil(tLeg);
    for (let k = 0; k < n; k++) {
      const h = Math.min(1, tLeg - k), mid = (k + 0.5 * h) * b.V_run_kn;
      steps.push([...destination(la, lo, bearing, mid), bearing, "steam", h, b.V_run_kn * KN]);
    }
    const [gla, glo] = destination(la, lo, bearing, dist);
    for (let k = 0; k < TRAWL_H; k++) {
      const hd = (bearing + (k % 2 === 0 ? 90 : -90) + 360) % 360;
      steps.push([gla, glo, hd, b.op_mode, 1.0, b.V_trawl_kn * KN]);
    }
    const back = (bearing + 180) % 360;
    for (let k = 0; k < n; k++) {
      const h = Math.min(1, tLeg - k), mid = dist - (k + 0.5 * h) * b.V_run_kn;
      steps.push([...destination(la, lo, bearing, mid), back, "steam", h, b.V_run_kn * KN]);
    }
    return steps.map(([lat, lon, hdg, mode, hours, V]) => ({ lat, lon, hdg: hdg * D2R, mode, hours, V }));
  }

  const rint = (x) => {               // numpy rint: round half to even
    const f = Math.floor(x), d = x - f;
    if (d > 0.5) return f + 1;
    if (d < 0.5) return f;
    return f % 2 === 0 ? f : f + 1;
  };

  /**
   * All (departure hour 02-08, speed 8/7/6 kn) options for one trip.
   * getWx(route, idxArray) -> {tws:[], twd:[] (rad, from), hs:[]} for each route step at forecast index idx.
   * dayStart = forecast index of 00:00 on the trip day.
   */
  function evaluate(boatKey, harbour, bearing, dist, sailKey, getWx, dayStart, nHours) {
    const rows = [];
    for (const v of [8.0, 7.0, 6.0]) {
      const b = makeBoat(boatKey, v), route = buildRoute(b, harbour, bearing, dist);
      const sail = SAIL_SETS[b.sail_set][sailKey], coeffs = sailCoeffs(sail);
      const offs = [0];
      for (let i = 0; i < route.length - 1; i++) offs.push(offs[i] + route[i].hours);
      for (let dep = 2; dep <= 8; dep++) {
        const idx = offs.map((o) => dayStart + dep + rint(o));
        if (Math.max(...idx) >= nHours) continue;
        const wx = getWx(route, idx, v);
        let f0 = 0, f1 = 0, sailH = 0, maxHs = 0, wsum = 0, tripH = 0;
        route.forEach((st, i) => {
          const tws = wx.tws[i], h = wx.hs[i];
          const [aws, awa] = apparentWind(tws, wx.twd[i], st.hdg, st.V);
          let net = sailNet(b, sail, coeffs, aws, awa, st.V, h, tws);
          if (b.op_mode === "drift" && st.mode === "drift") net = 0;
          f0 += fuelRate(b, st.mode, st.V, h, 0) * st.hours;
          f1 += fuelRate(b, st.mode, st.V, h, net) * st.hours;
          if (net > 0) sailH += st.hours;
          maxHs = Math.max(maxHs, h); wsum += tws; tripH += st.hours;
        });
        rows.push({ dep, speed: v, trip_h: tripH, fuel_nosail: f0, fuel_sail: f1, sail_hours: sailH,
          max_hs: maxHs, mean_wind: wsum / route.length });
      }
    }
    return rows;
  }

  const api = { KN, BOATS, SAIL_SETS, HARBOURS, makeBoat, buildRoute, apparentWind, sailNet, sailCoeffs,
    fuelRate, evaluate, rint };
  if (typeof module !== "undefined" && module.exports) module.exports = api; else root.VM = api;
})(typeof self !== "undefined" ? self : this);
