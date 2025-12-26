import argparse
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from itertools import product
import traceback

import numpy as np
import pandas as pd

from aircraft_config import AIRCRAFT_CONFIG
try:
    from aircraft_config import TURBOPROP_PARAMS
except ImportError:
    TURBOPROP_PARAMS = {}
from simulation import run_simulation

# Optional: PIL for assembling PDF summaries
try:
    from PIL import Image  # type: ignore
except Exception:
    Image = None  # type: ignore


def build_mach_grid(mmo: float) -> list[float]:
    # Start at MMO (inclusive) and step down by 0.01 to include values like 0.70
    # Keep within [0.30, MMO]
    start = max(0.30, min(round(float(mmo), 2), float(mmo)))
    grid = [round(max(0.30, start - 0.01 * i), 2) for i in range(8)]
    uniq = sorted(set([m for m in grid if 0.30 <= m <= float(mmo)]), reverse=True)
    return uniq


def build_alt_grid(ceiling_ft: int) -> list[int]:
    alts = list(range(int(ceiling_ft), 5000 - 1, -2000))
    if alts and alts[-1] != 5000:
        alts.append(5000)
    alts = [a for a in alts if a >= 5000]
    return alts


def payload_sweep(max_payload_lb: float, num_steps: int) -> list[int]:
    if max_payload_lb <= 0 or num_steps <= 0:
        return [0]
    if num_steps == 1:
        return [0]
    
    # Create equal steps from max_payload to 0
    step_size = max_payload_lb / (num_steps - 1)
    vals = []
    
    for i in range(num_steps):
        payload = round(max_payload_lb - i * step_size)
        if payload >= 0:  # Ensure we don't go negative due to rounding
            vals.append(payload)
    
    # Ensure we always include 0 as the last value
    if vals[-1] != 0:
        vals.append(0)
    
    return vals


def _write_plot_file(fig, out_path: Path, width: int, height: int, scale: int) -> tuple[bool, str | None, Path | None]:
    try:
        if fig is None or not hasattr(fig, "data") or fig.data is None or len(fig.data) == 0:
            return True, None, None
    except Exception:
        return True, None, None

    try:
        has_numeric_points = False
        for tr in fig.data:
            try:
                x = getattr(tr, "x", None)
                y = getattr(tr, "y", None)
                if x is None or y is None:
                    continue
                x_num = pd.to_numeric(pd.Series(np.array(x, dtype=object)), errors="coerce").to_numpy()
                y_num = pd.to_numeric(pd.Series(np.array(y, dtype=object)), errors="coerce").to_numpy()
                n = min(int(x_num.shape[0]), int(y_num.shape[0]))
                if n <= 0:
                    continue
                mask = np.isfinite(x_num[:n]) & np.isfinite(y_num[:n])
                if bool(np.any(mask)):
                    has_numeric_points = True
                    break
            except Exception:
                continue
        if not has_numeric_points:
            return True, None, None
    except Exception:
        return True, None, None
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    try:
        fig.write_image(str(out_path), width=int(width), height=int(height), scale=int(scale))
        return True, None, out_path
    except Exception as e:
        try:
            html_path = out_path.with_suffix(".html")
            fig.write_html(str(html_path), include_plotlyjs="cdn")
            return False, str(e), html_path
        except Exception:
            return False, str(e), None


def compute_initial_fuel(
    max_fuel: float,
    mrw: float,
    mtow: float,
    bow: float,
    payload: float,
    taxi_fuel: float,
) -> float:
    # Constrain fuel by:
    # - Tank capacity
    # - Ramp weight (MRW): bow + payload + fuel <= mrw
    # - Takeoff weight (MTOW): bow + payload + fuel - taxi_fuel <= mtow
    max_fuel_by_mrw = mrw - (bow + payload)
    max_fuel_by_mtow = mtow - (bow + payload) + taxi_fuel
    return float(max(0.0, min(max_fuel, max_fuel_by_mrw, max_fuel_by_mtow)))


def run_single_case(case: dict, hide_mach_limited: bool = False, hide_altitude_limited: bool = False) -> dict:
    try:
        (
            aircraft,
            mod,
            flap,
            isa_dev,
            cruise_alt,
            mach,
            payload,
            taxi_fuel,
            reserve_fuel,
        ) = (
            case["aircraft"],
            case["mod"],
            case["flap"],
            case["isa_dev"],
            case["cruise_alt"],
            case["mach"],
            case["payload"],
            case["taxi_fuel"],
            case["reserve_fuel"],
        )
        kias = case.get("kias")
        cruise_mode = case.get("cruise_mode")

        # Arbitrary route; range_mode ignores route distance triggers
        dep = "KSZT"
        arr = "KSAN"

        # Compute initial fuel respecting MRW and tank capacity
        ac = AIRCRAFT_CONFIG[(aircraft, mod)]
        bow = ac[18]
        max_fuel = ac[22]
        mrw = ac[20]
        mtow = ac[21]
        initial_fuel = compute_initial_fuel(max_fuel, mrw, mtow, bow, payload, taxi_fuel)

        if initial_fuel - reserve_fuel - taxi_fuel <= 0:
            return {
                **case,
                "status": "infeasible",
                "error_message": "No mission fuel (initial <= reserve + taxi)",
                "total_dist_nm": np.nan,
                "total_time_min": np.nan,
                "fuel_burned_lb": np.nan,
                "first_level_off_ft": np.nan,
                "cruise_vktas_kts": np.nan,
            }

        # Decide whether to pass explicit speed overrides
        _is_sweep_mode = (cruise_mode == "Speed Sweep (Mach/IAS)")
        cm_override = float(mach) if (_is_sweep_mode and kias is None) else None
        ck_override = float(kias) if (_is_sweep_mode and kias is not None) else None
        df, results, *_ = run_simulation(
            dep,
            arr,
            aircraft,
            mod,
            flap,
            payload,
            initial_fuel,
            taxi_fuel,
            reserve_fuel,
            int(cruise_alt),
            "No Wind",
            False,
            cruise_mach=cm_override,
            cruise_kias=ck_override,
            isa_dev_c=float(isa_dev),
            range_mode=True,
            cruise_mode=cruise_mode,
        )

        # If the sim returns constraint exceedances, treat the case as infeasible
        if isinstance(results, dict) and results.get("exceedances"):
            return {
                **case,
                "initial_fuel_lb": int(initial_fuel),
                "takeoff_weight_lb": int(bow + payload + initial_fuel - taxi_fuel),
                "status": "infeasible",
                "error_message": "; ".join([str(x) for x in results.get("exceedances", [])]),
                "total_dist_nm": np.nan,
                "total_time_min": np.nan,
                "fuel_burned_lb": np.nan,
                "first_level_off_ft": np.nan,
                "cruise_vktas_kts": np.nan,
                "cruise_vkias_kts": np.nan,
                "cruise_mode": cruise_mode,
            }

        # Extract outputs
        total_dist_nm = results.get("Total Dist (NM)")
        total_time_min = results.get("Total Time (min)")
        fuel_burned_lb = results.get("Total Fuel Burned (lb)")
        first_level_off_ft = results.get("First Level-Off Alt (ft)")
        
        # Compute representative cruise TAS (kts): highest achieved in cruise, prefer seg 7 then 6+7
        cruise_vktas_kts = None
        try:
            if isinstance(df, pd.DataFrame) and "VKTAS (kts)" in df.columns and "Segment" in df.columns:
                seg7_tas = pd.to_numeric(df.loc[df["Segment"] == 7, "VKTAS (kts)"] , errors="coerce")
                if seg7_tas.notna().any():
                    cruise_vktas_kts = float(np.nanmax(seg7_tas))
                else:
                    seg67_tas = pd.to_numeric(df.loc[df["Segment"].isin([6, 7]), "VKTAS (kts)"] , errors="coerce")
                    if seg67_tas.notna().any():
                        cruise_vktas_kts = float(np.nanmax(seg67_tas))
        except Exception:
            cruise_vktas_kts = None
        
        # Compute representative cruise IAS (kts): keep median for display stability
        cruise_vkias_kts = None
        try:
            if isinstance(df, pd.DataFrame) and "VKIAS (kts)" in df.columns and "Segment" in df.columns:
                seg7 = pd.to_numeric(df.loc[df["Segment"] == 7, "VKIAS (kts)"], errors="coerce")
                if seg7.notna().sum() > 0:
                    cruise_vkias_kts = float(np.nanmedian(seg7))
                else:
                    seg67 = pd.to_numeric(df.loc[df["Segment"].isin([6, 7]), "VKIAS (kts)"], errors="coerce")
                    if seg67.notna().sum() > 0:
                        cruise_vkias_kts = float(np.nanmedian(seg67))
        except Exception:
            cruise_vkias_kts = None

        # If time-history is available, compute fuel-limited range at reserve threshold
        try:
            if isinstance(df, pd.DataFrame) and "Fuel Remaining (lb)" in df.columns and "Distance (NM)" in df.columns:
                mask = df["Fuel Remaining (lb)"] >= float(reserve_fuel)
                if mask.any():
                    total_dist_nm = float(df.loc[mask, "Distance (NM)"].max())
                    # For reserve-limited range reporting, match burned fuel to mission fuel used to reach reserve
                    fuel_burned_lb = int(initial_fuel - taxi_fuel - reserve_fuel)
        except Exception:
            pass

        out = {
            **case,
            "initial_fuel_lb": int(initial_fuel),
            "takeoff_weight_lb": int(bow + payload + initial_fuel - taxi_fuel),
            "status": "ok" if not results.get("error") else "error",
            "error_message": results.get("error"),
            "total_dist_nm": total_dist_nm,
            "total_time_min": total_time_min,
            "fuel_burned_lb": fuel_burned_lb,
            "first_level_off_ft": first_level_off_ft,
            "cruise_vktas_kts": cruise_vktas_kts,
            "cruise_vkias_kts": cruise_vkias_kts,
            "cruise_mode": cruise_mode,
        }
        # Feasibility flags: altitude/mach limitations
        try:
            target_alt = int(cruise_alt)
            achieved_alt = int(first_level_off_ft) if first_level_off_ft is not None else None
            out["achieved_alt_ft"] = achieved_alt
            out["altitude_limited"] = achieved_alt is None or achieved_alt < target_alt
        except Exception:
            out["achieved_alt_ft"] = None
            out["altitude_limited"] = True
        try:
            # Compute achieved Mach from time history
            achieved_mach = None
            if isinstance(df, pd.DataFrame) and "Mach" in df.columns and "Segment" in df.columns:
                seg7_m = pd.to_numeric(df.loc[df["Segment"] == 7, "Mach"], errors="coerce")
                if seg7_m.notna().any():
                    achieved_mach = float(np.nanmax(seg7_m))
                else:
                    seg67_m = pd.to_numeric(df.loc[df["Segment"].isin([6, 7]), "Mach"], errors="coerce")
                    if seg67_m.notna().any():
                        achieved_mach = float(np.nanmax(seg67_m))
            out["achieved_mach"] = achieved_mach
            # If using optimization mode or MCT (no explicit target Mach), no Mach limitation
            if cruise_mode in ("Max Range", "Max Endurance", "MCT (Max Thrust)"):
                out["mach_limited"] = False
            else:
                target_mach = float(mach)
                out["mach_limited"] = achieved_mach is None or (achieved_mach + 1e-3) < target_mach
        except Exception:
            out["achieved_mach"] = None
            out["mach_limited"] = (cruise_mode not in ("Max Range", "Max Endurance"))
        
        # If hiding Mach-limited cases, replace numeric outputs with 'n/p'
        if hide_mach_limited and out.get("mach_limited", False):
            # Replace numeric result fields with 'n/p'
            for key in ["total_dist_nm", "total_time_min", "fuel_burned_lb", "first_level_off_ft", "cruise_vktas_kts", "cruise_vkias_kts"]:
                if key in out and out[key] is not None:
                    out[key] = "n/p"
            out["status"] = "mach_limited"
        
        # If hiding Altitude-limited cases, replace numeric outputs with 'n/p'
        if hide_altitude_limited and out.get("altitude_limited", False):
            # Replace numeric result fields with 'n/p'
            for key in ["total_dist_nm", "total_time_min", "fuel_burned_lb", "first_level_off_ft", "cruise_vktas_kts"]:
                if key in out and out[key] is not None:
                    out[key] = "n/p"
            out["status"] = "altitude_limited"
        # Save per-run PNG
        if case.get("save_plot") and results.get("fuel_distance_plot") is not None:
            try:
                fig = results["fuel_distance_plot"]
                plot_path: Path = case["plot_path"]
                ok, err, written_path = _write_plot_file(fig, plot_path, width=1600, height=900, scale=2)
                if (not ok) and err:
                    out["status"] = "plot_error" if out["status"] == "ok" else out["status"]
                    out["plot_error_message"] = f"{err}; wrote {str(written_path) if written_path else 'nothing'}"
            except Exception as e:
                out["status"] = "plot_error" if out["status"] == "ok" else out["status"]
                out["plot_error_message"] = str(e)
        # Save per-run timeseries
        if case.get("save_timeseries") and isinstance(df, pd.DataFrame) and not df.empty:
            try:
                ts_path: Path = case["timeseries_path"]
                ts_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_parquet(ts_path)
            except Exception as e:
                out["status"] = "ts_error" if out["status"] == "ok" else out["status"]
                out["timeseries_error_message"] = str(e)
        return out
    except Exception as e:
        return {
            **case,
            "status": "error",
            "error_message": str(e),
            "total_dist_nm": np.nan,
            "total_time_min": np.nan,
            "fuel_burned_lb": np.nan,
            "first_level_off_ft": np.nan,
            "cruise_vktas_kts": np.nan,
        }


def run_payload_range_batch(
    aircraft_models: list[str],
    mods: list[str],
    payload_steps: int = 6,
    taxi_fuel_lb: int = 100,
    isa_devs: list[int] = (-10, 0, 10, 20),
    flap_settings: list[int] = (0,),
    parallel_workers: int = 6,
    save_plots: bool = False,
    output_dir: str | Path | None = None,
    mach_values: list[float] | None = None,
    kias_values: list[float] | None = None,
    alt_values: list[int] | None = None,
    save_timeseries: bool = False,
    save_summary_plots: bool = True,
    hide_mach_limited: bool = False,
    hide_altitude_limited: bool = False,
    use_threads: bool = False,
    cruise_mode: str | None = None,
) -> pd.DataFrame:
    base_ts_dir = Path(output_dir) if output_dir else Path("batch_outputs") / datetime.now().strftime("%Y%m%d_%H%M%S")
    base_ts_dir.mkdir(parents=True, exist_ok=True)
    summary_plot_export_errors: list[str] = []
    
    # Create separate directories for each aircraft model
    aircraft_dirs = {}
    for aircraft in aircraft_models:
        aircraft_dir = base_ts_dir / aircraft
        aircraft_dir.mkdir(parents=True, exist_ok=True)
        aircraft_dirs[aircraft] = {
            "base": aircraft_dir,
            "plots": aircraft_dir / "plots",
            "summary_plots": aircraft_dir / "summary_plots",
            "timeseries": aircraft_dir / "timeseries"
        }
        aircraft_dirs[aircraft]["summary_plots"].mkdir(parents=True, exist_ok=True)

    # Build cases
    cases = []
    for aircraft in aircraft_models:
        for mod in mods:
            ac = AIRCRAFT_CONFIG.get((aircraft, mod))
            if not ac:
                continue
            ceiling = int(ac[8])
            mmo = float(ac[25])
            bow = float(ac[18])
            mzfw = float(ac[19])
            reserve_default = int(ac[24])
            max_payload = max(0.0, mzfw - bow)
            # Determine grids (use overrides if provided, filter by limits)
            if alt_values:
                alt_grid = sorted({int(a) for a in alt_values if 5000 <= int(a) <= int(ceiling)} , reverse=True)
                if not alt_grid:
                    alt_grid = build_alt_grid(ceiling)
            else:
                alt_grid = build_alt_grid(ceiling)
            # Determine speed grids
            is_optim_mode = (cruise_mode in ("Max Range", "Max Endurance"))
            is_sweep_mode = (cruise_mode == "Speed Sweep (Mach/IAS)")
            if not is_sweep_mode:
                # No speed sweep in MCT or optimized modes; simulator will set speed internally
                mach_grid = [0.0]
                kias_grid = None
            else:
                if mach_values:
                    mach_grid = sorted({float(mv) for mv in mach_values if 0.3 <= float(mv) <= float(mmo)} , reverse=True)
                    if not mach_grid:
                        mach_grid = build_mach_grid(mmo)
                else:
                    mach_grid = build_mach_grid(mmo)
                kias_grid = None
                is_turboprop = aircraft in TURBOPROP_PARAMS
                if is_turboprop and kias_values:
                    kias_grid = sorted({float(kv) for kv in kias_values if 60.0 <= float(kv) <= 300.0}, reverse=True)
                    if len(kias_grid) == 0:
                        kias_grid = None

            payloads = payload_sweep(max_payload, payload_steps)

            if kias_grid is not None:
                for flap, isa_dev, cruise_alt, kias, payload in product(
                    flap_settings, isa_devs, alt_grid, kias_grid, payloads
                ):
                    case = {
                        "aircraft": aircraft,
                        "mod": mod,
                        "flap": int(flap),
                        "isa_dev": int(isa_dev),
                        "cruise_alt": int(cruise_alt),
                        "mach": 0.0,
                        "kias": float(kias),
                        "payload": int(payload),
                        "taxi_fuel": int(taxi_fuel_lb),
                        "reserve_fuel": int(reserve_default),
                        "save_plot": bool(save_plots),
                        "cruise_mode": cruise_mode,
                    }
                    if save_plots:
                        plot_name = f"fuel_vs_distance_{aircraft}_{mod}_flap{flap}_kias{int(kias)}_alt{cruise_alt}_isa{isa_dev}_payload{payload}.png"
                        case["plot_path"] = aircraft_dirs[aircraft]["plots"] / plot_name
                    if save_timeseries:
                        ts_name = f"timeseries_{aircraft}_{mod}_flap{flap}_kias{int(kias)}_alt{cruise_alt}_isa{isa_dev}_payload{payload}.parquet"
                        case["timeseries_path"] = aircraft_dirs[aircraft]["timeseries"] / ts_name
                        case["save_timeseries"] = True
                    cases.append(case)
            else:
                for flap, isa_dev, cruise_alt, mach, payload in product(
                    flap_settings, isa_devs, alt_grid, mach_grid, payloads
                ):
                    case = {
                        "aircraft": aircraft,
                        "mod": mod,
                        "flap": int(flap),
                        "isa_dev": int(isa_dev),
                        "cruise_alt": int(cruise_alt),
                        "mach": float(mach),
                        "payload": int(payload),
                        "taxi_fuel": int(taxi_fuel_lb),
                        "reserve_fuel": int(reserve_default),
                        "save_plot": bool(save_plots),
                        "cruise_mode": cruise_mode,
                    }
                    if save_plots:
                        plot_name = f"fuel_vs_distance_{aircraft}_{mod}_flap{flap}_mach{mach:.2f}_alt{cruise_alt}_isa{isa_dev}_payload{payload}.png"
                        case["plot_path"] = aircraft_dirs[aircraft]["plots"] / plot_name
                    if save_timeseries:
                        ts_name = f"timeseries_{aircraft}_{mod}_flap{flap}_mach{mach:.2f}_alt{cruise_alt}_isa{isa_dev}_payload{payload}.parquet"
                        case["timeseries_path"] = aircraft_dirs[aircraft]["timeseries"] / ts_name
                        case["save_timeseries"] = True
                    cases.append(case)

    # Execute in parallel
    from functools import partial
    worker_func = partial(run_single_case, hide_mach_limited=hide_mach_limited, hide_altitude_limited=hide_altitude_limited)
    results = []
    Executor = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
    with Executor(max_workers=parallel_workers) as ex:
        fut_to_case = {ex.submit(worker_func, c): c for c in cases}
        for fut in as_completed(fut_to_case):
            case = fut_to_case[fut]
            try:
                results.append(fut.result())
            except Exception as e:
                results.append({**case, "status": "error", "error_message": str(e),
                                "total_dist_nm": np.nan, "total_time_min": np.nan, "fuel_burned_lb": np.nan,
                                "first_level_off_ft": np.nan, "cruise_vktas_kts": np.nan})

    df = pd.DataFrame(results)
    df["reserve_fuel_calc_lb"] = pd.to_numeric(df.get("initial_fuel_lb"), errors="coerce") - pd.to_numeric(df.get("fuel_burned_lb"), errors="coerce")

    # Save summary files by aircraft model
    for aircraft in aircraft_models:
        aircraft_df = df[df["aircraft"] == aircraft].copy()
        if len(aircraft_df) > 0:
            # Sort by mod (Flatwing, then Tamarack), altitude (desc), ISA (asc), Mach (desc), payload (asc)
            mod_order = pd.CategoricalDtype(["Flatwing", "Tamarack"], ordered=True)
            aircraft_df.loc[:, "mod"] = aircraft_df["mod"].astype(mod_order)
            aircraft_sorted = aircraft_df.sort_values(["mod","cruise_alt","isa_dev","mach","payload"], ascending=[True,False,True,False,True])
            aircraft_dir = aircraft_dirs[aircraft]["base"]
            
            # Round selected columns to 0 decimals for summary CSV (skip NA/inf)
            cols0 = ["total_dist_nm", "cruise_vkias_kts"]
            display_df = aircraft_sorted.copy()
            for _c in cols0:
                if _c in display_df.columns:
                    _num = pd.to_numeric(display_df[_c], errors="coerce")
                    rounded = _num.round(0)
                    finite_mask = pd.Series(np.isfinite(rounded.values), index=rounded.index)
                    mask = rounded.notna() & finite_mask
                    display_df.loc[mask, _c] = rounded[mask].astype(int)
            
            # Save CSV and parquet
            display_df.to_csv(aircraft_dir / "summary.csv", index=False)
            try:
                aircraft_sorted.to_parquet(aircraft_dir / "summary.parquet", index=False)
            except Exception:
                pass
    
    # Also save combined summary in base directory (sorted by aircraft model first, then mod, alt desc, ISA asc, Mach desc, payload asc)
    df_sorted = df.copy()
    mod_order = pd.CategoricalDtype(["Flatwing", "Tamarack"], ordered=True)
    df_sorted.loc[:, "mod"] = df_sorted["mod"].astype(mod_order)
    df_sorted = df_sorted.sort_values(["aircraft","mod","cruise_alt","isa_dev","mach","payload"], ascending=[True,True,False,True,False,True])
    # Round selected columns to 0 decimals for combined summary CSV (skip NA/inf)
    cols0 = ["total_dist_nm", "cruise_vkias_kts"]
    display_df2 = df_sorted.copy()
    for _c in cols0:
        if _c in display_df2.columns:
            _num = pd.to_numeric(display_df2[_c], errors="coerce")
            rounded = _num.round(0)
            finite_mask = pd.Series(np.isfinite(rounded.values), index=rounded.index)
            mask = rounded.notna() & finite_mask
            display_df2.loc[mask, _c] = rounded[mask].astype(int)
    display_df2.to_csv(base_ts_dir / "combined_summary.csv", index=False)
    try:
        df_sorted.to_parquet(base_ts_dir / "combined_summary.parquet", index=False)
    except Exception:
        pass

    # Derived tables (long form) - save by aircraft and combined
    try:
        # Save comprehensive CSVs by aircraft
        for aircraft in aircraft_models:
            aircraft_df = df[df["aircraft"] == aircraft].copy()
            if len(aircraft_df) > 0:
                aircraft_dir = aircraft_dirs[aircraft]["base"]
                
                # Payload range all data for this aircraft
                df_aircraft_all = aircraft_df[[
                    "aircraft","mod","flap","isa_dev","cruise_alt","mach","payload",
                    "total_dist_nm","total_time_min","fuel_burned_lb","first_level_off_ft","cruise_vktas_kts","cruise_vkias_kts"
                ]].copy().sort_values(["mod","cruise_alt","mach","payload"], ascending=[True,False,False,True])
                df_aircraft_all.to_csv(aircraft_dir / "payload_range_all.csv", index=False)
                
                # Payload=0 data for this aircraft
                df_aircraft_payload0 = aircraft_df[aircraft_df["payload"] == 0].copy()
                if len(df_aircraft_payload0) > 0:
                    # Range vs Mach
                    df_rvm = df_aircraft_payload0[["aircraft","mod","flap","isa_dev","cruise_alt","mach","payload","total_dist_nm"]].dropna()
                    df_rvm.to_csv(aircraft_dir / "range_vs_mach_payload0.csv", index=False)
                    # Endurance vs Mach
                    df_evm = df_aircraft_payload0[["aircraft","mod","flap","isa_dev","cruise_alt","mach","payload","total_time_min"]].dropna()
                    df_evm.to_csv(aircraft_dir / "endurance_vs_mach_payload0.csv", index=False)
                    # Range vs Altitude
                    df_rva = df_aircraft_payload0[["aircraft","mod","flap","isa_dev","mach","cruise_alt","payload","total_dist_nm"]].dropna()
                    df_rva.to_csv(aircraft_dir / "range_vs_altitude_payload0.csv", index=False)
        
        # Also save combined files in base directory
        df_all = df[[
            "aircraft","mod","flap","isa_dev","cruise_alt","mach","payload",
            "total_dist_nm","total_time_min","fuel_burned_lb","first_level_off_ft","cruise_vktas_kts","cruise_vkias_kts"
        ]].copy().sort_values(["aircraft","mod","cruise_alt","mach","payload"], ascending=[True,True,False,False,True])
        df_all.to_csv(base_ts_dir / "combined_payload_range_all.csv", index=False)

        df_payload0 = df[df["payload"] == 0].copy()
        if len(df_payload0) > 0:
            df_rvm = df_payload0[["aircraft","mod","flap","isa_dev","cruise_alt","mach","payload","total_dist_nm"]].dropna()
            df_rvm.to_csv(base_ts_dir / "combined_range_vs_mach_payload0.csv", index=False)
            df_evm = df_payload0[["aircraft","mod","flap","isa_dev","cruise_alt","mach","payload","total_time_min"]].dropna()
            df_evm.to_csv(base_ts_dir / "combined_endurance_vs_mach_payload0.csv", index=False)
            df_rva = df_payload0[["aircraft","mod","flap","isa_dev","mach","cruise_alt","payload","total_dist_nm"]].dropna()
            df_rva.to_csv(base_ts_dir / "combined_range_vs_altitude_payload0.csv", index=False)
    except Exception:
        pass

    # Save meta files - one combined and one per aircraft
    combined_meta = {
        "aircraft_models": aircraft_models,
        "mods": mods,
        "payload_steps": payload_steps,
        "taxi_fuel_lb": taxi_fuel_lb,
        "isa_devs": list(isa_devs),
        "flap_settings": list(flap_settings),
        "save_plots": save_plots,
        "save_summary_plots": save_summary_plots,
        "save_timeseries": save_timeseries,
        "parallel_workers": parallel_workers,
        "output_dir": str(base_ts_dir),
        "run_type": "combined_multi_aircraft"
    }
    pd.Series(combined_meta).to_json(base_ts_dir / "meta.json")
    
    # Save individual meta files for each aircraft
    for aircraft in aircraft_models:
        aircraft_meta = {
            "aircraft_models": [aircraft],
            "mods": mods,
            "payload_steps": payload_steps,
            "taxi_fuel_lb": taxi_fuel_lb,
            "isa_devs": list(isa_devs),
            "flap_settings": list(flap_settings),
            "save_plots": save_plots,
            "save_summary_plots": save_summary_plots,
            "save_timeseries": save_timeseries,
            "parallel_workers": parallel_workers,
            "output_dir": str(aircraft_dirs[aircraft]["base"]),
            "run_type": "individual_aircraft"
        }
        pd.Series(aircraft_meta).to_json(aircraft_dirs[aircraft]["base"] / "meta.json")

    # Generate summary plots (PNG) - save to respective aircraft directories
    if save_summary_plots:
        try:
            import plotly.graph_objects as go
            # Payload-Range overlays
            for aircraft in sorted(df["aircraft"].dropna().unique()):
                df_a = df[df["aircraft"]==aircraft]
                aircraft_summary_dir = aircraft_dirs[aircraft]["summary_plots"]
                try:
                    aircraft_summary_dir.mkdir(parents=True, exist_ok=True)
                except Exception:
                    pass

                for alt in sorted(df_a["cruise_alt"].dropna().unique(), reverse=True):
                    if cruise_mode in ("Max Range", "Max Endurance"):
                        # Optimized modes: single plot per altitude (no Mach facet)
                        df_filtered = df_a[df_a["cruise_alt"]==alt]
                        if "altitude_limited" in df_filtered.columns:
                            df_filtered = df_filtered[df_filtered["altitude_limited"] == False]
                        if df_filtered.empty:
                            continue
                        df_filtered = df_filtered.copy()
                        df_filtered.loc[:, "isa_label"] = df_filtered["isa_dev"].apply(lambda x: f"ISA {x:+d}°C")
                        fig = go.Figure()
                        for mod in sorted(df_filtered["mod"].unique()):
                            df_mod = df_filtered[df_filtered["mod"] == mod]
                            if df_mod.empty:
                                continue
                            color = "red" if mod == "Flatwing" else "green"
                            label_mod = "BL" if mod == "Flatwing" else "Mod"
                            for isa_dev in sorted(df_mod["isa_dev"].unique()):
                                df_isa = df_mod[df_mod["isa_dev"] == isa_dev]
                                if df_isa.empty:
                                    continue
                                df_plot = df_isa.sort_values("payload", ascending=False).copy()
                                isa_label = f"ISA {isa_dev:+d}°C"
                                try:
                                    att_m = pd.to_numeric(df_isa.get("achieved_mach"), errors="coerce").dropna()
                                    att_m_txt = f"M {att_m.max():.2f}" if len(att_m) > 0 else "M —"
                                except Exception:
                                    att_m_txt = "M —"
                                try:
                                    att_alt = pd.to_numeric(df_isa.get("first_level_off_ft"), errors="coerce").dropna()
                                    att_fl_txt = f"FL{int(att_alt.max()/100)}" if len(att_alt) > 0 else "FL—"
                                except Exception:
                                    att_fl_txt = "FL—"
                                # Choose X-axis: Range for Max Range, Endurance for Max Endurance
                                if cruise_mode == "Max Endurance":
                                    df_plot.loc[:, "x_num"] = pd.to_numeric(df_plot["total_time_min"], errors="coerce")
                                    x_label = "Endurance (min)"
                                    title_prefix = "Payload-Endurance"
                                else:
                                    df_plot.loc[:, "x_num"] = pd.to_numeric(df_plot["total_dist_nm"], errors="coerce")
                                    x_label = "Range (NM)"
                                    title_prefix = "Payload-Range"
                                df_plot = df_plot.dropna(subset=["x_num", "payload"])    
                                if df_plot.empty:
                                    continue
                                df_plot = df_plot.groupby("payload", as_index=False).agg(x_num=("x_num", "max"))
                                df_plot = df_plot.sort_values("payload", ascending=False)
                                x_vals = df_plot["x_num"].to_numpy()
                                y_vals = df_plot["payload"].to_numpy()
                                x_mon = np.maximum.accumulate(x_vals)
                                # Do not prepend a zero anchor; use actual computed values
                                x_plot = x_mon
                                if len(x_plot) > 1:
                                    keep = np.concatenate(([True], x_plot[1:] > x_plot[:-1] + 1e-9))
                                    x_plot = x_plot[keep]
                                    y_vals = y_vals[keep]
                                fig.add_trace(go.Scatter(
                                    x=x_plot,
                                    y=y_vals,
                                    mode="lines+markers",
                                    name=f"{label_mod} Max Achieved {att_m_txt}, Attained {att_fl_txt} - {isa_label}",
                                    line=dict(color=color, dash=("longdashdot" if isa_dev == -20 else "dash" if isa_dev == -10 else "solid" if isa_dev == 0 else "dot" if isa_dev == 10 else "dashdot" if isa_dev == 20 else "longdash"), width=2.5),
                                    marker=dict(symbol=(
                                        "triangle-up" if isa_dev == -20 else
                                        "square" if isa_dev == -10 else
                                        "circle" if isa_dev == 0 else
                                        "diamond" if isa_dev == 10 else
                                        "x" if isa_dev == 20 else
                                        "triangle-down"
                                    ), size=6)
                                ))
                                if len(x_plot) > 0:
                                    fig.add_trace(go.Scatter(
                                        x=[0, float(x_plot[0])],
                                        y=[float(y_vals[0]), float(y_vals[0])],
                                        mode="lines",
                                        line=dict(color=color, width=2.0),
                                        showlegend=False
                                    ))
                        title_mode = "Max Endurance" if cruise_mode == "Max Endurance" else "Max Range"
                        fig.update_layout(
                            title=f"{title_prefix} | {aircraft} | FL{int(alt/100)} Goal | Optimized for {title_mode}",
                            xaxis_title=x_label,
                            yaxis_title="Payload (lb)",
                            template="plotly_white"
                        )
                        fig.update_xaxes(rangemode="tozero")
                        fname = (
                            aircraft_summary_dir / f"payload_endurance_{aircraft}_alt{int(alt)}_optimized.png"
                            if cruise_mode == "Max Endurance" else
                            aircraft_summary_dir / f"payload_range_{aircraft}_alt{int(alt)}_optimized.png"
                        )
                        ok, err, written_path = _write_plot_file(fig, fname, width=1400, height=900, scale=2)
                        if (not ok) and err:
                            summary_plot_export_errors.append(
                                f"{aircraft} alt={int(alt)} mode=optimized target={fname.name}: {err}; wrote {written_path.name if written_path else 'nothing'}"
                            )
                    elif cruise_mode == "Speed Sweep (Mach/IAS)":
                        # Speed Sweep: facet by Mach as before
                        for mach in sorted(df_a["mach"].dropna().unique(), reverse=True):
                            df_filtered = df_a[(df_a["cruise_alt"]==alt) & (np.isclose(df_a["mach"], mach))]
                            if df_filtered.empty:
                                continue
                            df_filtered = df_filtered.copy()
                            df_filtered.loc[:, "isa_label"] = df_filtered["isa_dev"].apply(lambda x: f"ISA {x:+d}°C")
                            fig = go.Figure()
                            for mod in sorted(df_filtered["mod"].unique()):
                                df_mod = df_filtered[df_filtered["mod"] == mod]
                                if df_mod.empty:
                                    continue
                                color = "red" if mod == "Flatwing" else "green"
                                label_mod = "BL" if mod == "Flatwing" else "Mod"
                                for isa_dev in sorted(df_mod["isa_dev"].unique()):
                                    df_isa = df_mod[df_mod["isa_dev"] == isa_dev]
                                    if df_isa.empty:
                                        continue
                                    df_plot = df_isa.sort_values("payload", ascending=False).copy()
                                    isa_label = f"ISA {isa_dev:+d}°C"
                                    try:
                                        att_m = pd.to_numeric(df_isa.get("achieved_mach"), errors="coerce").dropna()
                                        att_m_txt = f"M {att_m.max():.2f}" if len(att_m) > 0 else "M —"
                                    except Exception:
                                        att_m_txt = "M —"
                                    try:
                                        att_alt = pd.to_numeric(df_isa.get("first_level_off_ft"), errors="coerce").dropna()
                                        att_fl_txt = f"FL{int(att_alt.max()/100)}" if len(att_alt) > 0 else "FL—"
                                    except Exception:
                                        att_fl_txt = "FL—"
                                    df_plot.loc[:, "total_dist_nm_num"] = pd.to_numeric(df_plot["total_dist_nm"], errors="coerce")
                                    df_plot = df_plot.dropna(subset=["total_dist_nm_num", "payload"])    
                                    if df_plot.empty:
                                        continue
                                    df_plot = df_plot.groupby("payload", as_index=False).agg(total_dist_nm_num=("total_dist_nm_num", "max"))
                                    df_plot = df_plot.sort_values("payload", ascending=False)
                                    x_vals = df_plot["total_dist_nm_num"].to_numpy()
                                    y_vals = df_plot["payload"].to_numpy()
                                    x_mon = np.maximum.accumulate(x_vals)
                                    # Do not prepend a zero anchor; use actual computed values
                                    x_plot = x_mon
                                    if len(x_plot) > 1:
                                        keep = np.concatenate(([True], x_plot[1:] > x_plot[:-1] + 1e-9))
                                        x_plot = x_plot[keep]
                                        y_vals = y_vals[keep]
                                    fig.add_trace(go.Scatter(
                                        x=x_plot,
                                        y=y_vals,
                                        mode="lines",
                                        name=f"{label_mod} Max Achieved {att_m_txt}, Attained {att_fl_txt} - {isa_label}",
                                        line=dict(color=color, dash=("dash" if isa_dev == -10 else "solid" if isa_dev == 0 else "dot" if isa_dev == 10 else "dashdot"))
                                    ))
                                    if len(x_plot) > 0:
                                        fig.add_trace(go.Scatter(
                                            x=[0, float(x_plot[0])],
                                            y=[float(y_vals[0]), float(y_vals[0])],
                                            mode="lines",
                                            line=dict(color=color, width=2.0),
                                            showlegend=False
                                        ))
                            fig.update_layout(
                                title=f"Payload-Range | {aircraft} | FL{int(alt/100)} Goal | M {mach:.2f} Goal",
                                xaxis_title="Range (NM)",
                                yaxis_title="Payload (lb)",
                                template="plotly_white"
                            )
                            fig.update_xaxes(rangemode="tozero")
                            fname = aircraft_summary_dir / f"payload_range_{aircraft}_alt{int(alt)}_mach{mach:.2f}.png"
                            ok, err, written_path = _write_plot_file(fig, fname, width=1400, height=900, scale=2)
                            if (not ok) and err:
                                summary_plot_export_errors.append(
                                    f"{aircraft} alt={int(alt)} mach={float(mach):.2f} target={fname.name}: {err}; wrote {written_path.name if written_path else 'nothing'}"
                                )
                    else:
                        # MCT: no speed facet; mirror optimized behavior but without 'Optimized' in filename
                        df_filtered = df_a[df_a["cruise_alt"]==alt]
                        if df_filtered.empty:
                            continue
                        df_filtered = df_filtered.copy()
                        df_filtered.loc[:, "isa_label"] = df_filtered["isa_dev"].apply(lambda x: f"ISA {x:+d}°C")
                        fig = go.Figure()
                        for mod in sorted(df_filtered["mod"].unique()):
                            df_mod = df_filtered[df_filtered["mod"] == mod]
                            if df_mod.empty:
                                continue
                            color = "red" if mod == "Flatwing" else "green"
                            label_mod = "BL" if mod == "Flatwing" else "Mod"
                            for isa_dev in sorted(df_mod["isa_dev"].unique()):
                                df_isa = df_mod[df_mod["isa_dev"] == isa_dev]
                                if df_isa.empty:
                                    continue
                                df_plot = df_isa.sort_values("payload", ascending=False).copy()
                                isa_label = f"ISA {isa_dev:+d}°C"
                                try:
                                    att_m = pd.to_numeric(df_isa.get("achieved_mach"), errors="coerce").dropna()
                                    att_m_txt = f"M {att_m.max():.2f}" if len(att_m) > 0 else "M —"
                                except Exception:
                                    att_m_txt = "M —"
                                try:
                                    att_alt = pd.to_numeric(df_isa.get("first_level_off_ft"), errors="coerce").dropna()
                                    att_fl_txt = f"FL{int(att_alt.max()/100)}" if len(att_alt) > 0 else "FL—"
                                except Exception:
                                    att_fl_txt = "FL—"
                                df_plot.loc[:, "total_dist_nm_num"] = pd.to_numeric(df_plot["total_dist_nm"], errors="coerce")
                                df_plot = df_plot.dropna(subset=["total_dist_nm_num", "payload"])    
                                if df_plot.empty:
                                    continue
                                df_plot = df_plot.groupby("payload", as_index=False).agg(total_dist_nm_num=("total_dist_nm_num", "max"))
                                df_plot = df_plot.sort_values("payload", ascending=False)
                                x_vals = df_plot["total_dist_nm_num"].to_numpy()
                                y_vals = df_plot["payload"].to_numpy()
                                x_mon = np.maximum.accumulate(x_vals)
                                x_plot = x_mon
                                if len(x_plot) > 1:
                                    keep = np.concatenate(([True], x_plot[1:] > x_plot[:-1] + 1e-9))
                                    x_plot = x_plot[keep]
                                    y_vals = y_vals[keep]
                                fig.add_trace(go.Scatter(
                                    x=x_plot,
                                    y=y_vals,
                                    mode="lines+markers",
                                    name=f"{label_mod} Max Achieved {att_m_txt}, Attained {att_fl_txt} - {isa_label}",
                                    line=dict(color=color, dash=("longdashdot" if isa_dev == -20 else "dash" if isa_dev == -10 else "solid" if isa_dev == 0 else "dot" if isa_dev == 10 else "dashdot" if isa_dev == 20 else "longdash"), width=2.5),
                                    marker=dict(symbol=(
                                        "triangle-up" if isa_dev == -20 else
                                        "square" if isa_dev == -10 else
                                        "circle" if isa_dev == 0 else
                                        "diamond" if isa_dev == 10 else
                                        "x" if isa_dev == 20 else
                                        "triangle-down"
                                    ), size=6)
                                ))
                                if len(x_plot) > 0:
                                    fig.add_trace(go.Scatter(
                                        x=[0, float(x_plot[0])],
                                        y=[float(y_vals[0]), float(y_vals[0])],
                                        mode="lines",
                                        line=dict(color=color, width=2.0),
                                        showlegend=False
                                    ))
                        fig.update_layout(
                            title=f"Payload-Range | {aircraft} | FL{int(alt/100)} Goal | MCT (Max Thrust)",
                            xaxis_title="Range (NM)",
                            yaxis_title="Payload (lb)",
                            template="plotly_white"
                        )
                        fig.update_xaxes(rangemode="tozero")
                        fname = aircraft_summary_dir / f"payload_range_{aircraft}_alt{int(alt)}_mct.png"
                        ok, err, written_path = _write_plot_file(fig, fname, width=1400, height=900, scale=2)
                        if (not ok) and err:
                            summary_plot_export_errors.append(
                                f"{aircraft} alt={int(alt)} mode=mct target={fname.name}: {err}; wrote {written_path.name if written_path else 'nothing'}"
                            )
            # MCT mode: Save both Range vs Altitude and Endurance vs Altitude (payload=0)
            if cruise_mode == "MCT (Max Thrust)":
                for aircraft in sorted(df["aircraft"].dropna().unique()):
                    df_a0 = df[(df["aircraft"]==aircraft) & (df["payload"]==0)]
                    if df_a0.empty:
                        continue
                    aircraft_summary_dir = aircraft_dirs[aircraft]["summary_plots"]
                    # Common prep
                    df_alt = df_a0.copy()
                    df_alt.loc[:, "isa_label"] = df_alt["isa_dev"].apply(lambda x: f"ISA {x:+d}°C")
                    # Helper to build a figure given y column and labels
                    import plotly.graph_objects as go
                    def _plot_altitude_family(y_col: str, y_label: str, title_txt: str, out_name: str):
                        fig = go.Figure()
                        for mod in sorted(df_alt["mod"].dropna().unique().tolist()):
                            d_mod = df_alt[df_alt["mod"] == mod]
                            for dev in sorted(pd.to_numeric(d_mod["isa_dev"], errors="coerce").dropna().unique().tolist()):
                                d_dev = d_mod[pd.to_numeric(d_mod["isa_dev"], errors="coerce") == dev]
                                d_x = pd.to_numeric(d_dev.get("cruise_alt"), errors="coerce")
                                d_y = pd.to_numeric(d_dev.get(y_col), errors="coerce")
                                try:
                                    d_xy = pd.DataFrame({"x": d_x, "y": d_y}).dropna()
                                except Exception:
                                    d_xy = pd.DataFrame()
                                if d_xy is None or d_xy.empty:
                                    continue
                                try:
                                    att_m = pd.to_numeric(d_dev.get("achieved_mach"), errors="coerce").dropna()
                                    att_m_txt = f"M {att_m.max():.2f}" if len(att_m) > 0 else "M —"
                                except Exception:
                                    att_m_txt = "M —"
                                try:
                                    att_alt = pd.to_numeric(d_dev.get("first_level_off_ft"), errors="coerce").dropna()
                                    att_fl_txt = f"FL{int(att_alt.max()/100)}" if len(att_alt) > 0 else "FL—"
                                except Exception:
                                    att_fl_txt = "FL—"
                                color = "red" if mod == "Flatwing" else "green"
                                fig.add_trace(go.Scatter(
                                    x=d_xy["x"],
                                    y=d_xy["y"],
                                    mode="lines+markers",
                                    name=f"{'BL' if mod=='Flatwing' else 'Mod'} Max Achieved {att_m_txt}, Attained {att_fl_txt} - ISA {int(dev):+d}°C",
                                    line=dict(color=color, dash=(
                                        "longdashdot" if int(dev) == -20 else
                                        "dash" if int(dev) == -10 else
                                        "solid" if int(dev) == 0 else
                                        "dot" if int(dev) == 10 else
                                        "dashdot" if int(dev) == 20 else
                                        "longdash"
                                    ), width=2.5),
                                    marker=dict(symbol=(
                                        "triangle-up" if int(dev) == -20 else
                                        "square" if int(dev) == -10 else
                                        "circle" if int(dev) == 0 else
                                        "diamond" if int(dev) == 10 else
                                        "x" if int(dev) == 20 else
                                        "triangle-down"
                                    ), size=6)
                                ))
                        fig.update_layout(title=title_txt, xaxis_title="Target Altitude (ft)", yaxis_title=y_label, template="plotly_white")
                        if fig.data and len(fig.data) > 0:
                            fname = aircraft_summary_dir / out_name
                            ok, err, written_path = _write_plot_file(fig, fname, width=1600, height=900, scale=2)
                            if (not ok) and err:
                                summary_plot_export_errors.append(
                                    f"{aircraft} mode=mct family target={fname.name}: {err}; wrote {written_path.name if written_path else 'nothing'}"
                                )
                    # Range vs Altitude (MCT)
                    _plot_altitude_family(
                        y_col="total_dist_nm",
                        y_label="Max Range (NM)",
                        title_txt=f"Range vs Altitude (MCT) | {aircraft}",
                        out_name=f"range_vs_altitude_{aircraft}_mct.png",
                    )
                    # Endurance vs Altitude (MCT)
                    _plot_altitude_family(
                        y_col="total_time_min",
                        y_label="Max Endurance (min)",
                        title_txt=f"Endurance vs Altitude (MCT) | {aircraft}",
                        out_name=f"endurance_vs_altitude_{aircraft}_mct.png",
                    )
            # Optimized modes: Save BOTH Range vs Altitude and Endurance vs Altitude (payload=0)
            if cruise_mode in ("Max Range", "Max Endurance"):
                for aircraft in sorted(df["aircraft"].dropna().unique()):
                    df_a0 = df[(df["aircraft"]==aircraft) & (df["payload"]==0)]
                    if df_a0.empty:
                        continue
                    aircraft_summary_dir = aircraft_dirs[aircraft]["summary_plots"]
                    df_alt = df_a0.copy()
                    df_alt.loc[:, "isa_label"] = df_alt["isa_dev"].apply(lambda x: f"ISA {x:+d}°C")
                    import plotly.graph_objects as go
                    def _plot_opt_family(y_col: str, y_label: str, title_txt: str, out_name: str):
                        fig = go.Figure()
                        for mod in sorted(df_alt["mod"].dropna().unique().tolist()):
                            d_mod = df_alt[df_alt["mod"] == mod]
                            for dev in sorted(pd.to_numeric(d_mod["isa_dev"], errors="coerce").dropna().unique().tolist()):
                                d_dev = d_mod[pd.to_numeric(d_mod["isa_dev"], errors="coerce") == dev]
                                d_x = pd.to_numeric(d_dev.get("cruise_alt"), errors="coerce")
                                d_y = pd.to_numeric(d_dev.get(y_col), errors="coerce")
                                try:
                                    d_xy = pd.DataFrame({"x": d_x, "y": d_y}).dropna()
                                except Exception:
                                    d_xy = pd.DataFrame()
                                if d_xy is None or d_xy.empty:
                                    continue

                                try:
                                    att_m = pd.to_numeric(d_dev.get("achieved_mach"), errors="coerce").dropna()
                                    att_m_txt = f"M {att_m.max():.2f}" if len(att_m) > 0 else "M —"
                                except Exception:
                                    att_m_txt = "M —"
                                try:
                                    att_alt = pd.to_numeric(d_dev.get("first_level_off_ft"), errors="coerce").dropna()
                                    att_fl_txt = f"FL{int(att_alt.max()/100)}" if len(att_alt) > 0 else "FL—"
                                except Exception:
                                    att_fl_txt = "FL—"

                                color = "red" if mod == "Flatwing" else "green"
                                fig.add_trace(go.Scatter(
                                    x=d_xy["x"],
                                    y=d_xy["y"],
                                    mode="lines+markers",
                                    name=f"{'BL' if mod=='Flatwing' else 'Mod'} Max Achieved {att_m_txt}, Attained {att_fl_txt} - ISA {int(dev):+d}°C",
                                    line=dict(color=color, dash=(
                                        "longdashdot" if int(dev) == -20 else
                                        "dash" if int(dev) == -10 else
                                        "solid" if int(dev) == 0 else
                                        "dot" if int(dev) == 10 else
                                        "dashdot" if int(dev) == 20 else
                                        "longdash"
                                    ), width=2.5),
                                    marker=dict(symbol=(
                                        "triangle-up" if int(dev) == -20 else
                                        "square" if int(dev) == -10 else
                                        "circle" if int(dev) == 0 else
                                        "diamond" if int(dev) == 10 else
                                        "x" if int(dev) == 20 else
                                        "triangle-down"
                                    ), size=6)
                                ))

                        fig.update_layout(title=title_txt, xaxis_title="Target Altitude (ft)", yaxis_title=y_label, template="plotly_white")
                        if fig.data and len(fig.data) > 0:
                            fname = aircraft_summary_dir / out_name
                            ok, err, written_path = _write_plot_file(fig, fname, width=1600, height=900, scale=2)
                            if (not ok) and err:
                                summary_plot_export_errors.append(
                                    f"{aircraft} mode=opt family target={fname.name}: {err}; wrote {written_path.name if written_path else 'nothing'}"
                                )
                    # Range vs Altitude (Optimized)
                    _plot_opt_family(
                        y_col="total_dist_nm",
                        y_label="Max Range (NM)",
                        title_txt=f"Range vs Altitude (Optimized) | {aircraft}",
                        out_name=f"range_vs_altitude_{aircraft}_optimized.png",
                    )
                    # Endurance vs Altitude (Optimized)
                    _plot_opt_family(
                        y_col="total_time_min",
                        y_label="Max Endurance (min)",
                        title_txt=f"Endurance vs Altitude (Optimized) | {aircraft}",
                        out_name=f"endurance_vs_altitude_{aircraft}_optimized.png",
                    )
            # Family: Range vs Mach by Altitude (payload 0) with ISA temperature separation
            if cruise_mode == "Speed Sweep (Mach/IAS)":
                for aircraft in sorted(df["aircraft"].dropna().unique()):
                    df_a0 = df[(df["aircraft"]==aircraft) & (df["payload"]==0)]
                    aircraft_summary_dir = aircraft_dirs[aircraft]["summary_plots"]
                    
                    for alt in sorted(df_a0["cruise_alt"].dropna().unique(), reverse=True):
                        df_alt = df_a0[df_a0["cruise_alt"] == alt]
                        if df_alt.empty:
                            continue
                        
                        # Create ISA deviation labels for better legend
                        df_alt = df_alt.copy()
                        df_alt.loc[:, "isa_label"] = df_alt["isa_dev"].apply(lambda x: f"ISA {x:+d}°C")
                        
                        fig = go.Figure()
                        for mod in sorted(df_alt["mod"].unique()):
                            df_mod = df_alt[df_alt["mod"] == mod]
                            if df_mod.empty:
                                continue
                            
                            color = "red" if mod == "Flatwing" else "green"
                            label_mod = "BL" if mod == "Flatwing" else "Mod"
                            
                            for isa_dev in sorted(df_mod["isa_dev"].unique()):
                                df_isa = df_mod[df_mod["isa_dev"] == isa_dev]
                                if df_isa.empty:
                                    continue
                                
                                df_plot = df_isa.sort_values("mach")
                                isa_label = f"ISA {isa_dev:+d}°C"
                                
                                # Legend: show attained first level-off altitude instead of target altitude
                                try:
                                    att_alt = pd.to_numeric(df_plot.get("first_level_off_ft"), errors="coerce").dropna()
                                    att_fl_txt = f"FL{int(att_alt.max()/100)}" if len(att_alt) > 0 else "FL—"
                                except Exception:
                                    att_fl_txt = "FL—"
                                fig.add_trace(go.Scatter(
                                    x=df_plot["mach"], 
                                    y=df_plot["total_dist_nm"], 
                                    mode="lines", 
                                    name=f"{label_mod} Attained {att_fl_txt} - {isa_label}",
                                    line=dict(color=color, dash=("dash" if isa_dev == -10 else "solid" if isa_dev == 0 else "dot" if isa_dev == 10 else "dashdot"))
                                ))
                        
                        fig.update_layout(
                            title=f"Range vs Mach (Payload=0) | {aircraft} | FL{int(alt/100)} Goal",
                            xaxis_title="Target Mach",
                            yaxis_title="Range (NM)",
                            template="plotly_white"
                        )
                        fname = aircraft_summary_dir / f"range_vs_mach_{aircraft}_FL{int(alt/100)}.png"
                        ok, err, written_path = _write_plot_file(fig, fname, width=1600, height=900, scale=2)
                        if (not ok) and err:
                            summary_plot_export_errors.append(
                                f"{aircraft} family range_vs_mach target={fname.name}: {err}; wrote {written_path.name if written_path else 'nothing'}"
                            )
            if cruise_mode == "Speed Sweep (Mach/IAS)":
                # Family: Endurance vs Mach by Altitude (payload 0)
                for aircraft in sorted(df["aircraft"].dropna().unique()):
                    df_a0 = df[(df["aircraft"]==aircraft) & (df["payload"]==0)]
                    aircraft_summary_dir = aircraft_dirs[aircraft]["summary_plots"]
                    
                    for alt in sorted(df_a0["cruise_alt"].dropna().unique(), reverse=True):
                        df_alt = df_a0[df_a0["cruise_alt"] == alt]
                        if df_alt.empty:
                            continue
                        
                        df_alt = df_alt.copy()
                        df_alt.loc[:, "isa_label"] = df_alt["isa_dev"].apply(lambda x: f"ISA {x:+d}°C")
                        fig = go.Figure()
                        for mod in sorted(df_alt["mod"].unique()):
                            df_mod = df_alt[df_alt["mod"] == mod]
                            if df_mod.empty:
                                continue
                            color = "red" if mod == "Flatwing" else "green"
                            label_mod = "BL" if mod == "Flatwing" else "Mod"
                            for isa_dev in sorted(df_mod["isa_dev"].unique()):
                                df_isa = df_mod[df_mod["isa_dev"] == isa_dev]
                                if df_isa.empty:
                                    continue
                                df_plot = df_isa.sort_values("mach")
                                isa_label = f"ISA {isa_dev:+d}°C"
                                try:
                                    att_alt = pd.to_numeric(df_plot.get("first_level_off_ft"), errors="coerce").dropna()
                                    att_fl_txt = f"FL{int(att_alt.max()/100)}" if len(att_alt) > 0 else "FL—"
                                except Exception:
                                    att_fl_txt = "FL—"
                                fig.add_trace(go.Scatter(
                                    x=df_plot["mach"], 
                                    y=df_plot["total_time_min"], 
                                    mode="lines", 
                                    name=f"{label_mod} Attained {att_fl_txt} - {isa_label}",
                                    line=dict(color=color, dash=("dash" if isa_dev == -10 else "solid" if isa_dev == 0 else "dot" if isa_dev == 10 else "dashdot"))
                                ))
                        fig.update_layout(
                            title=f"Endurance vs Mach (Payload=0) | {aircraft} | FL{int(alt/100)} Goal",
                            xaxis_title="Target Mach",
                            yaxis_title="Endurance (min)",
                            template="plotly_white"
                        )
                        fname = aircraft_summary_dir / f"endurance_vs_mach_{aircraft}_FL{int(alt/100)}.png"
                        ok, err, written_path = _write_plot_file(fig, fname, width=1600, height=900, scale=2)
                        if (not ok) and err:
                            summary_plot_export_errors.append(
                                f"{aircraft} family endurance_vs_mach target={fname.name}: {err}; wrote {written_path.name if written_path else 'nothing'}"
                            )
            
            if cruise_mode == "Speed Sweep (Mach/IAS)":
                # Family: Range vs Altitude by Mach (payload 0)
                for aircraft in sorted(df["aircraft"].dropna().unique()):
                    df_a0 = df[(df["aircraft"]==aircraft) & (df["payload"]==0)]
                    aircraft_summary_dir = aircraft_dirs[aircraft]["summary_plots"]
                    
                    for mach in sorted(df_a0["mach"].dropna().unique(), reverse=True):
                        df_mach = df_a0[np.isclose(df_a0["mach"], mach)]
                        if df_mach.empty:
                            continue
                        df_mach = df_mach.copy()
                        df_mach.loc[:, "isa_label"] = df_mach["isa_dev"].apply(lambda x: f"ISA {x:+d}°C")
                        fig = go.Figure()
                        for mod in sorted(df_mach["mod"].unique()):
                            df_mod = df_mach[df_mach["mod"] == mod]
                            if df_mod.empty:
                                continue
                            color = "red" if mod == "Flatwing" else "green"
                            label_mod = "BL" if mod == "Flatwing" else "Mod"
                            for isa_dev in sorted(df_mod["isa_dev"].unique()):
                                df_isa = df_mod[df_mod["isa_dev"] == isa_dev]
                                if df_isa.empty:
                                    continue
                                df_plot = df_isa.sort_values("cruise_alt")
                                isa_label = f"ISA {isa_dev:+d}°C"
                                try:
                                    att_m = pd.to_numeric(df_plot.get("achieved_mach"), errors="coerce").dropna()
                                    att_m_txt = f"M {att_m.max():.2f}" if len(att_m) > 0 else "M —"
                                except Exception:
                                    att_m_txt = "M —"
                                try:
                                    att_alt = pd.to_numeric(df_plot.get("first_level_off_ft"), errors="coerce").dropna()
                                    att_fl_txt = f"FL{int(att_alt.max()/100)}" if len(att_alt) > 0 else "FL—"
                                except Exception:
                                    att_fl_txt = "FL—"
                                fig.add_trace(go.Scatter(
                                    x=df_plot["cruise_alt"], 
                                    y=df_plot["total_dist_nm"], 
                                    mode="lines", 
                                    name=f"{label_mod} Max Achieved {att_m_txt}, Attained {att_fl_txt} - {isa_label}",
                                    line=dict(color=color, dash=("dash" if isa_dev == -10 else "solid" if isa_dev == 0 else "dot" if isa_dev == 10 else "dashdot"))
                                ))
                        fig.update_layout(
                            title=f"Range vs Altitude (Payload=0) | {aircraft} | M {mach:.2f} Goal",
                            xaxis_title="Target Altitude (ft)",
                            yaxis_title="Range (NM)",
                            template="plotly_white"
                        )
                        fname = aircraft_summary_dir / f"range_vs_altitude_{aircraft}_M{mach:.2f}.png"
                        ok, err, written_path = _write_plot_file(fig, fname, width=1600, height=900, scale=2)
                        if (not ok) and err:
                            summary_plot_export_errors.append(
                                f"{aircraft} family range_vs_altitude target={fname.name}: {err}; wrote {written_path.name if written_path else 'nothing'}"
                            )
            if cruise_mode == "Speed Sweep (Mach/IAS)":
                # Family: Endurance vs Altitude by Mach (payload 0)
                for aircraft in sorted(df["aircraft"].dropna().unique()):
                    df_a0 = df[(df["aircraft"]==aircraft) & (df["payload"]==0)]
                    aircraft_summary_dir = aircraft_dirs[aircraft]["summary_plots"]
                    
                    for mach in sorted(df_a0["mach"].dropna().unique(), reverse=True):
                        df_mach = df_a0[np.isclose(df_a0["mach"], mach)]
                        if df_mach.empty:
                            continue
                        df_mach = df_mach.copy()
                        df_mach.loc[:, "isa_label"] = df_mach["isa_dev"].apply(lambda x: f"ISA {x:+d}°C")
                        fig = go.Figure()
                        for mod in sorted(df_mach["mod"].unique()):
                            df_mod = df_mach[df_mach["mod"] == mod]
                            if df_mod.empty:
                                continue
                            color = "red" if mod == "Flatwing" else "green"
                            label_mod = "BL" if mod == "Flatwing" else "Mod"
                            for isa_dev in sorted(df_mod["isa_dev"].unique()):
                                df_isa = df_mod[df_mod["isa_dev"] == isa_dev]
                                if df_isa.empty:
                                    continue
                                df_plot = df_isa.sort_values("cruise_alt")
                                isa_label = f"ISA {isa_dev:+d}°C"
                                try:
                                    att_m = pd.to_numeric(df_plot.get("achieved_mach"), errors="coerce").dropna()
                                    att_m_txt = f"M {att_m.max():.2f}" if len(att_m) > 0 else "M —"
                                except Exception:
                                    att_m_txt = "M —"
                                try:
                                    att_alt = pd.to_numeric(df_plot.get("first_level_off_ft"), errors="coerce").dropna()
                                    att_fl_txt = f"FL{int(att_alt.max()/100)}" if len(att_alt) > 0 else "FL—"
                                except Exception:
                                    att_fl_txt = "FL—"
                                fig.add_trace(go.Scatter(
                                    x=df_plot["cruise_alt"], 
                                    y=df_plot["total_time_min"], 
                                    mode="lines", 
                                    name=f"{label_mod} Max Achieved {att_m_txt}, Attained {att_fl_txt} - {isa_label}",
                                    line=dict(color=color, dash=("dash" if isa_dev == -10 else "solid" if isa_dev == 0 else "dot" if isa_dev == 10 else "dashdot"))
                                ))
                        fig.update_layout(
                            title=f"Endurance vs Altitude (Payload=0) | {aircraft} | M {mach:.2f} Goal",
                            xaxis_title="Target Altitude (ft)",
                            yaxis_title="Endurance (min)",
                            template="plotly_white"
                        )
                        fname = aircraft_summary_dir / f"endurance_vs_altitude_{aircraft}_M{mach:.2f}.png"
                        ok, err, written_path = _write_plot_file(fig, fname, width=1600, height=900, scale=2)
                        if (not ok) and err:
                            summary_plot_export_errors.append(
                                f"{aircraft} family endurance_vs_altitude target={fname.name}: {err}; wrote {written_path.name if written_path else 'nothing'}"
                            )
        except Exception:
            # Best-effort; ignore plotting errors to keep batch running
            summary_plot_export_errors.append(str(traceback.format_exc()))
        try:
            if summary_plot_export_errors:
                (base_ts_dir / "summary_plot_export_errors.txt").write_text("\n".join(summary_plot_export_errors), encoding="utf-8")
        except Exception:
            pass
        # Assemble a single PDF per-aircraft in the model folder from summary_plots PNGs
        try:
            if Image is not None:
                for aircraft in sorted(df["aircraft"].dropna().unique()):
                    aircraft_dir = aircraft_dirs[aircraft]["base"]
                    summary_dir = aircraft_dirs[aircraft]["summary_plots"]
                    png_files = sorted(summary_dir.glob("*.png"))
                    if not png_files:
                        continue
                    images = []
                    for p in png_files:
                        try:
                            im = Image.open(p).convert("RGB")
                            images.append(im)
                        except Exception:
                            continue
                    if images:
                        pdf_path = aircraft_dir / "summary_plots.pdf"
                        first, rest = images[0], images[1:]
                        first.save(pdf_path, save_all=True, append_images=rest)
        except Exception:
            pass

    try:
        df.attrs["output_dir"] = str(base_ts_dir)
        df.attrs["summary_plot_export_error_count"] = int(len(summary_plot_export_errors))
    except Exception:
        pass
    return df


def parse_args():
    p = argparse.ArgumentParser(description="Batch payload-range sweep runner")
    p.add_argument("--aircraft", nargs="+", required=True, help="Aircraft models, e.g., CJ1 M2 CJ2")
    p.add_argument("--mods", nargs="+", default=["Flatwing", "Tamarack"], help="Mods to include")
    p.add_argument("--payload-step", type=int, default=200)
    p.add_argument("--taxi-fuel", type=int, default=100)
    p.add_argument("--isa", nargs="+", type=int, default=[-10, 0, 10, 20])
    p.add_argument("--flaps", nargs="+", type=int, default=[0])
    p.add_argument("--parallel", type=int, default=6)
    p.add_argument("--mach", nargs="*", type=float, default=None, help="Optional custom Mach values (e.g., --mach 0.70 0.69 0.68)")
    p.add_argument("--kias", nargs="*", type=float, default=None, help="Optional IAS values for turboprops in kts (e.g., --kias 170 160 150)")
    p.add_argument("--alts", nargs="*", type=int, default=None, help="Optional custom altitudes in ft (e.g., --alts 41000 39000 35000)")
    p.add_argument("--no-plots", action="store_true", help="Disable per-run PNG plot export (fuel vs distance)")
    p.add_argument("--no-summary-plots", action="store_true", help="Disable summary PNG plots (payload-range and family plots)")
    p.add_argument("--out", type=str, default=None, help="Output directory (default batch_outputs/{timestamp})")
    return p.parse_args()


def main():
    args = parse_args()
    run_payload_range_batch(
        aircraft_models=args.aircraft,
        mods=args.mods,
        payload_steps=args.payload_step,
        taxi_fuel_lb=args.taxi_fuel,
        isa_devs=args.isa,
        flap_settings=args.flaps,
        parallel_workers=args.parallel,
        save_plots=not args.no_plots,
        output_dir=args.out,
        mach_values=args.mach,
        kias_values=args.kias,
        alt_values=args.alts,
        save_summary_plots=not args.no_summary_plots,
    )


if __name__ == "__main__":
    main()
