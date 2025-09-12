# generate_wind_weeks.py
# Create four seasonal weekly wind capacity-factor profiles (WINTER, SPRING, SUMMER, AUTUMN)
# for N wind buses, using a Weibull marginal + AR(1) temporal correlation + mild diurnal cycle.
# Outputs individual CSVs per bus/season, combined per-season CSVs, and master long/wide tables.

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# -----------------------------
# Math helpers (no scipy needed)
# -----------------------------
def erf_approx(x: np.ndarray) -> np.ndarray:
    """Vectorized erf approximation (Abramowitz & Stegun 7.1.26)."""
    a1=0.254829592; a2=-0.284496736; a3=1.421413741
    a4=-1.453152027; a5=1.061405429; p=0.3275911
    sign = np.sign(x)
    x = np.abs(x)
    t = 1.0/(1.0 + p*x)
    y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*np.exp(-x*x)
    return sign * y

def norm_cdf(x: np.ndarray) -> np.ndarray:
    return 0.5*(1.0 + erf_approx(x/np.sqrt(2.0)))

def weibull_inverse(u: np.ndarray, k: float, c: float) -> np.ndarray:
    u = np.clip(u, 1e-12, 1-1e-12)
    return c * (-np.log(1 - u))**(1.0/k)

def ar1_gaussian(hours: int, rho: float, rng: np.random.Generator) -> np.ndarray:
    x = np.zeros(hours)
    eps = rng.normal(size=hours)
    for t in range(1, hours):
        x[t] = rho * x[t-1] + np.sqrt(1 - rho**2) * eps[t]
    return x

def speed_to_cf(v: np.ndarray, v_ci=3.5, v_r=12.0, v_co=25.0) -> np.ndarray:
    v = np.asarray(v)
    cf = np.zeros_like(v)
    mask_ramp = (v >= v_ci) & (v < v_r)
    mask_rated = (v >= v_r) & (v < v_co)
    if np.any(mask_ramp):
        num = v[mask_ramp]**3 - v_ci**3
        den = max(v_r**3 - v_ci**3, 1e-12)
        cf[mask_ramp] = num / den
    cf[mask_rated] = 1.0
    return cf

def make_week_cf(hours=168, k=2.0, c=9.0, rho=0.85, seed=0,
                 diurnal_amp=0.10, avail_losses=0.92, CF_target=None,
                 v_ci=3.5, v_r=12.0, v_co=25.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = ar1_gaussian(hours, rho, rng)     # AR(1) Gaussian
    u = norm_cdf(x)                        # → U(0,1)
    v = weibull_inverse(u, k, c)           # Weibull inverse CDF → wind speed
    # Mild diurnal modulation (peak near 15:00)
    h = np.arange(hours) % 24
    diurnal = 1.0 + diurnal_amp * np.sin(2*np.pi*(h - 15)/24.0)
    v = v * diurnal
    # Speed → capacity factor via simple turbine curve
    cf = speed_to_cf(v, v_ci=v_ci, v_r=v_r, v_co=v_co)
    cf = cf * avail_losses
    # Optional mean CF calibration
    if CF_target is not None:
        m = cf.mean()
        if m > 1e-8:
            cf = np.minimum(1.0, cf * (CF_target / m))
    return cf

# -----------------------------
# Main generator
# -----------------------------
def generate_profiles(outdir: Path,
                      buses: list[int],
                      start_date: str = "2025-01-01",
                      hours: int = 24*7,
                      rho: float = 0.85,
                      diurnal_amp: float = 0.10,
                      avail_losses: float = 0.92,
                      v_ci: float = 3.5, v_r: float = 12.0, v_co: float = 25.0):
    outdir.mkdir(parents=True, exist_ok=True)

    # Four seasonal configs (tweak as needed)
    seasons = [
        {"name": "WINTER", "k": 2.0, "c": 8.0, "CF_target": 0.35},
        {"name": "SPRING", "k": 2.1, "c": 9.0, "CF_target": 0.40},
        {"name": "SUMMER", "k": 1.9, "c": 7.5, "CF_target": 0.30},
        {"name": "AUTUMN", "k": 2.2, "c": 9.5, "CF_target": 0.42},
    ]

    # Default per-bus seeds (different base seeds → different but comparable profiles)
    seed_bases = {b: (1000 + 137*b) for b in buses}

    all_rows = []
    for s_idx, sc in enumerate(seasons):
        name, k, c, CF_target = sc["name"], sc["k"], sc["c"], sc["CF_target"]
        # Distinct timestamps per season (7-day offsets)
        time_index = pd.date_range(start_date, periods=hours, freq="H") + pd.Timedelta(days=7*s_idx)
        df_week = pd.DataFrame({"timestamp": time_index})
        for b in buses:
            seed = seed_bases[b] + s_idx
            cf = make_week_cf(hours=hours, k=k, c=c, rho=rho, seed=seed,
                              diurnal_amp=diurnal_amp, avail_losses=avail_losses,
                              CF_target=CF_target, v_ci=v_ci, v_r=v_r, v_co=v_co)
            col = f"CF_bus{b}_{name}"
            df_week[col] = cf
            # Per-bus/season CSV
            pd.DataFrame({"timestamp": time_index, "CF": cf}).to_csv(
                outdir / f"wind_CF_bus{b}_{name}.csv", index=False
            )
        # Combined (all buses) for the season
        df_week.to_csv(outdir / f"wind_CF_buses_{name}.csv", index=False)
        # Long-format slice
        df_melt = df_week.melt(id_vars=["timestamp"], var_name="series", value_name="CF")
        df_melt.insert(1, "season", name)
        all_rows.append(df_melt)

    # Master tables
    master = pd.concat(all_rows, ignore_index=True)
    master.to_csv(outdir / "wind_CF_master_long.csv", index=False)

    # Wide panel (timestamps as rows, columns CF_bus{b}_{SEASON})
    wide = master.pivot_table(index="timestamp", columns="series", values="CF").reset_index()
    # Reorder columns
    cols = ["timestamp"]
    for sc in seasons:
        for b in buses:
            cols.append(f"CF_bus{b}_{sc['name']}")
    wide = wide[cols]
    wide.to_csv(outdir / "wind_CF_panel_wide.csv", index=False)

    # Quick summary print
    print(f"[OK] Wrote CSVs to: {outdir.resolve()}")
    for sc in seasons:
        print(f"  - Season: {sc['name']}")
        for b in buses:
            fn = outdir / f"wind_CF_bus{b}_{sc['name']}.csv"
            print(f"      bus {b}: {fn.name}")

# -----------------------------
# CLI
# -----------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Generate seasonal weekly wind CF profiles for wind buses.")
    ap.add_argument("--out", type=Path, default=Path("./wind_profiles"), help="Output directory")
    ap.add_argument("--buses", type=int, nargs="+", default=[2, 3], help="Wind bus IDs (e.g., 2 3)")
    ap.add_argument("--start-date", type=str, default="2025-01-01", help="Start date for timestamps")
    ap.add_argument("--rho", type=float, default=0.85, help="AR(1) correlation (0.0–0.99)")
    ap.add_argument("--diurnal", type=float, default=0.10, help="Diurnal amplitude (0.0–0.5)")
    ap.add_argument("--avail", type=float, default=0.92, help="Availability & losses multiplier")
    ap.add_argument("--vci", type=float, default=3.5, help="Cut-in wind speed (m/s)")
    ap.add_argument("--vr", type=float, default=12.0, help="Rated wind speed (m/s)")
    ap.add_argument("--vco", type=float, default=25.0, help="Cut-out wind speed (m/s)")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    generate_profiles(outdir=args.out,
                      buses=args.buses,
                      start_date=args.start_date,
                      rho=args.rho,
                      diurnal_amp=args.diurnal,
                      avail_losses=args.avail,
                      v_ci=args.vci, v_r=args.vr, v_co=args.vco)
