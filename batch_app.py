import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.express as px
import plotly.graph_objects as go

from aircraft_config import AIRCRAFT_CONFIG
try:
    from aircraft_config import TURBOPROP_PARAMS
except ImportError:
    TURBOPROP_PARAMS = {}
from batch.payload_range import run_payload_range_batch

# Detect Kaleido for static image export
try:
    import kaleido  # type: ignore
    _kaleido_available = True
except Exception:
    _kaleido_available = False

st.set_page_config(page_title="Payload–Range Sweeps", layout="wide")
st.title("Batch Payload–Range Sweeps")

# Sidebar configuration
with st.sidebar:
    st.header("Configuration")

    # Aircraft selection
    aircraft_types = sorted({a for a, _ in AIRCRAFT_CONFIG.keys()})
    selected_aircraft = st.multiselect("Aircraft Models", options=aircraft_types, default=["CJ1"])  # default CJ1

    # Mods selection
    mods_all = ["Flatwing", "Tamarack"]
    selected_mods = st.multiselect("Mods", options=mods_all, default=mods_all)

    # Flaps - hardcoded to 0 (zero flaps)
    selected_flaps = [0]

    # ISA deviations
    isa_default = [-10, 0, 10, 20]
    selected_isa = st.multiselect("ISA Deviations (C)", options=[-20, -10, 0, 10, 20, 30], default=isa_default)

    # Policy
    payload_steps = st.number_input("Number of Payload Steps (from max to zero)", min_value=1, max_value=20, step=1, value=6)
    taxi_fuel = st.number_input("Taxi Fuel (lb)", min_value=0, max_value=1000, step=10, value=100)

    # Performance
    parallel = st.number_input("Parallel Workers", min_value=1, max_value=16, step=1, value=6)
    runs_per_min = st.number_input(
        "Assumed throughput (runs/min)", min_value=1, max_value=2000, step=1,
        value=int(st.session_state.get("assumed_rpm", 30))
    )

    # Cruise mode selection (applies to cruise segment above 10k ft)
    cruise_mode = st.radio(
        "Cruise Mode",
        ["MCT (Max Thrust)", "Max Range", "Max Endurance"],
        index=0,
        help="For Max Range/Endurance, cruise Mach is optimized using CL/CD criteria with V >= 1.2 × Vs (EAS)."
    )

    # Output options
    output_mode = st.radio(
        "Output Mode",
        options=["Output All Data", "Hide Mach-Limited Cases"],
        index=0,
        help="'Hide Mach-Limited Cases' will not output data for cases where achieved Mach < target Mach and will show 'n/p' in tables."
    )
    altitude_mode = st.radio(
        "Altitude Output Mode",
        options=["Output All Data", "Hide Altitude-Limited Cases"],
        index=0,
        help="'Hide Altitude-Limited Cases' will not output data for cases where achieved altitude < target altitude and will show 'n/p' in tables."
    )

    # Sweep grids (range with steps)
    st.header("Sweep Grids")
    st.caption("Custom ranges will be filtered per-aircraft (Mach <= MMO, Alt <= ceiling, Alt >= 5000). Use negative steps for descending ranges.")
    if cruise_mode == "MCT (Max Thrust)":
        # Prefer IAS for turboprops
        selected_is_turboprop = bool(selected_aircraft) and all(a in TURBOPROP_PARAMS for a in selected_aircraft)
        speed_type = st.radio("Speed type", options=["Mach", "IAS (kts)"], index=(1 if selected_is_turboprop else 0), horizontal=True)
        if speed_type == "Mach":
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                mach_start = st.number_input("Mach start", value=0.70, step=0.01, format="%.2f")
            with col_m2:
                mach_end = st.number_input("Mach end", value=0.62, step=0.01, format="%.2f")
            with col_m3:
                mach_step = st.number_input("Mach step", value=-0.01, step=0.01, format="%.2f")
            kias_start = kias_end = kias_step = None
        else:
            col_k1, col_k2, col_k3 = st.columns(3)
            with col_k1:
                kias_start = st.number_input("IAS start (kts)", value=170, step=5)
            with col_k2:
                kias_end = st.number_input("IAS end (kts)", value=150, step=5)
            with col_k3:
                kias_step = st.number_input("IAS step (kts)", value=-10, step=1)
            mach_start = mach_end = mach_step = None
    else:
        # Optimization modes: no speed sweep
        speed_type = None
        mach_start = mach_end = mach_step = None
        kias_start = kias_end = kias_step = None

    col_a1, col_a2, col_a3 = st.columns(3)
    with col_a1:
        alt_start = st.number_input("Altitude start (ft)", value=41000, step=1000)
    with col_a2:
        alt_end = st.number_input("Altitude end (ft)", value=5000, step=1000)
    with col_a3:
        alt_step = st.number_input("Altitude step (ft)", value=-2000, step=500)

    # Plot types
    st.header("Plots to show")
    payload_curve_label = ("Payload-Endurance Curves" if cruise_mode == "Max Endurance" else "Payload-Range Curves")
    if cruise_mode == "MCT (Max Thrust)":
        plot_options = [
            payload_curve_label,
            "Max Range vs Mach",
            "Max Endurance vs Mach",
            "Max Range vs Altitude",
            "Endurance vs Altitude",
        ]
        default_plots = [payload_curve_label, "Max Range vs Mach", "Max Endurance vs Mach", "Max Range vs Altitude", "Endurance vs Altitude"]
        plots_key = "plots_mct"
    else:
        # Optimized modes: hide Mach-based plots entirely and only show the relevant altitude plot
        if cruise_mode == "Max Endurance":
            plot_options = [
                payload_curve_label,
                "Endurance vs Altitude",
            ]
            default_plots = [payload_curve_label, "Endurance vs Altitude"]
            plots_key = "plots_endurance"
        else:
            plot_options = [
                payload_curve_label,
                "Max Range vs Altitude",
            ]
            default_plots = [payload_curve_label, "Max Range vs Altitude"]
            plots_key = "plots_range"
    plot_choices = st.multiselect("Select plots", options=plot_options, default=default_plots, key=plots_key)
    save_summary_plots_ui = st.checkbox(
        "Save summary plots (PNG)", value=_kaleido_available,
        help="Writes static images to each aircraft's summary_plots folder"
    )
    if save_summary_plots_ui and not _kaleido_available:
        st.info("Kaleido not installed; disabling summary plot export.")
        save_summary_plots_ui = False

    # Rough ETA estimate based on current inputs (ignores per-aircraft ceiling/MMO filtering)
    def _count_range_float(start: float, end: float, step: float) -> int:
        try:
            if step == 0:
                return 1
            if step > 0:
                if end < start:
                    return 1
                return int(np.floor((end - start) / step)) + 1
            else:
                if start < end:
                    return 1
                return int(np.floor((start - end) / (-step))) + 1
        except Exception:
            return 1
    def _count_range_int(start: int, end: int, step: int) -> int:
        try:
            if step == 0:
                return 1
            if step > 0:
                if end < start:
                    return 1
                return (end - start) // step + 1
            else:
                if start < end:
                    return 1
                return (start - end) // (-step) + 1
        except Exception:
            return 1

    if cruise_mode == "MCT (Max Thrust)":
        if speed_type == "Mach":
            speed_count_est = _count_range_float(float(mach_start), float(mach_end), float(mach_step))
        else:
            speed_count_est = _count_range_int(int(kias_start), int(kias_end), int(kias_step))
    else:
        speed_count_est = 1
    alt_count_est = _count_range_int(int(alt_start), int(alt_end), int(alt_step))
    payload_count_est = int(payload_steps)
    est_total_runs = max(1, len(selected_aircraft) * len(selected_mods) * len(selected_isa) * 1 * speed_count_est * alt_count_est * payload_count_est)
    eta_min_est = est_total_runs / max(1, int(runs_per_min))
    st.caption(f"Estimated ~{est_total_runs} runs; ETA ≈ {eta_min_est:.1f} min at {int(runs_per_min)} runs/min")

# Run button
run_clicked = st.button("Run Sweep", type="primary")

if run_clicked:
    if not selected_aircraft:
        st.warning("Select at least one aircraft.")
        st.stop()
    if not selected_mods:
        st.warning("Select at least one mod.")
        st.stop()

    with st.spinner("Running sweeps. This can take a while..."):
        # Build ranges from start/end/step
        def build_range(start: float, end: float, step: float) -> list[float]:
            vals: list[float] = []
            if step == 0:
                return [round(float(start), 3)]
            x = float(start)
            if step > 0:
                while x <= float(end) + 1e-9:
                    vals.append(round(x, 3))
                    x += step
            else:
                while x >= float(end) - 1e-9:
                    vals.append(round(x, 3))
                    x += step
            # unique preserve order
            out: list[float] = []
            seen: set[float] = set()
            for v in vals:
                if v not in seen:
                    out.append(v)
                    seen.add(v)
            return out

        def build_int_range(start: int, end: int, step: int) -> list[int]:
            vals: list[int] = []
            if step == 0:
                return [int(start)]
            x = int(start)
            if step > 0:
                while x <= int(end):
                    vals.append(int(x))
                    x += step
            else:
                while x >= int(end):
                    vals.append(int(x))
                    x += step
            return list(dict.fromkeys(vals))

        if cruise_mode == "MCT (Max Thrust)":
            if mach_start is not None:
                mach_values = build_range(float(mach_start), float(mach_end), float(mach_step))
                kias_values = None
            else:
                mach_values = None
                kias_values = build_int_range(int(kias_start), int(kias_end), int(kias_step))
        else:
            # Optimization modes: no speed grid
            mach_values = None
            kias_values = None
        alt_values = build_int_range(int(alt_start), int(alt_end), int(alt_step))

        t0 = time.perf_counter()
        df = run_payload_range_batch(
            aircraft_models=selected_aircraft,
            mods=selected_mods,
            payload_steps=int(payload_steps),
            taxi_fuel_lb=int(taxi_fuel),
            isa_devs=[int(x) for x in selected_isa],
            flap_settings=[0],
            parallel_workers=int(parallel),
            save_plots=False,
            save_summary_plots=save_summary_plots_ui,
            mach_values=mach_values,
            kias_values=kias_values,
            alt_values=alt_values,
            hide_mach_limited=(output_mode == "Hide Mach-Limited Cases"),
            hide_altitude_limited=(altitude_mode == "Hide Altitude-Limited Cases"),
            use_threads=True,
            cruise_mode=cruise_mode,
        )
        elapsed_sec = time.perf_counter() - t0
    st.session_state.batch_summary = df
    st.session_state.batch_elapsed_sec = elapsed_sec
    if len(df) > 0 and elapsed_sec > 0:
        st.session_state.assumed_rpm = max(1, int(len(df) / (elapsed_sec / 60.0)))

# Results area
summary_df: pd.DataFrame | None = st.session_state.get("batch_summary")
if summary_df is not None and len(summary_df) > 0:
    elapsed_sec = st.session_state.get("batch_elapsed_sec", None)
    if elapsed_sec is not None and elapsed_sec > 0:
        minutes = elapsed_sec / 60.0
        rpm = (len(summary_df) / minutes) if minutes > 0 else 0.0
        st.success(f"Completed {len(summary_df)} runs in {minutes:.1f} minutes ({rpm:.1f} runs/min)")
    else:
        st.success(f"Completed {len(summary_df)} runs")

    # Downloads
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button(
            label="Download summary.csv",
            data=summary_df.to_csv(index=False).encode("utf-8"),
            file_name="summary.csv",
            mime="text/csv",
        )
    with col_dl2:
        try:
            import io
            import pyarrow as pa  # type: ignore
            import pyarrow.parquet as pq  # type: ignore
            buf = io.BytesIO()
            table = pa.Table.from_pandas(summary_df)
            pq.write_table(table, buf)
            st.download_button(
                label="Download summary.parquet",
                data=buf.getvalue(),
                file_name="summary.parquet",
                mime="application/octet-stream",
            )
        except Exception:
            pass

    st.markdown("---")

    # Filter widgets for plotting
    plot_aircraft = st.multiselect("Filter: Aircraft", options=sorted(summary_df["aircraft"].unique().tolist()), default=sorted(summary_df["aircraft"].unique().tolist()))
    plot_mods = st.multiselect("Filter: Mods", options=sorted(summary_df["mod"].unique().tolist()), default=sorted(summary_df["mod"].unique().tolist()))
    # Flaps filter - removed since only flap=0 is used
    plot_isa = st.multiselect("Filter: ISA Dev (C)", options=sorted(summary_df["isa_dev"].unique().tolist()), default=sorted(summary_df["isa_dev"].unique().tolist()))

    filtered = summary_df[
        summary_df["aircraft"].isin(plot_aircraft)
        & summary_df["mod"].isin(plot_mods)
        & summary_df["isa_dev"].isin(plot_isa)
    ].copy()
    # Determine whether this dataset contains IAS (turboprops) or Mach (jets)
    # Presence of a numeric 'kias' column indicates IAS mode
    has_kias_global = ("kias" in filtered.columns) and pd.to_numeric(filtered.get("kias"), errors="coerce").notna().any()
    speed_col_global = "kias" if has_kias_global else "mach"

    # Payload curves (Range or Endurance depending on mode)
    if payload_curve_label in plot_choices:
        st.subheader("Payload-Endurance Curves" if cruise_mode == "Max Endurance" else "Payload-Range Curves")
        # Select altitude (speed selection optional depending on cruise mode)
        alt_options = sorted(filtered["cruise_alt"].unique().tolist(), reverse=True)
        sel_alt = st.selectbox("Cruise Altitude", options=alt_options) if alt_options else None
        if sel_alt is not None:
            if cruise_mode == "MCT (Max Thrust)":
                # Keep speed facet for MCT
                has_kias = ("kias" in filtered.columns) and pd.to_numeric(filtered.get("kias"), errors="coerce").notna().any()
                speed_col = "kias" if has_kias else "mach"
                speed_label = "IAS (kts)" if speed_col == "kias" else "Mach"
                speed_options = sorted(pd.to_numeric(filtered[speed_col], errors="coerce").dropna().unique().tolist(), reverse=True)
                sel_speed = st.selectbox(speed_label, options=speed_options) if speed_options else None
                df_pr = filtered[(filtered["cruise_alt"] == sel_alt) & (
                    (np.isclose(pd.to_numeric(filtered["mach"], errors="coerce"), float(sel_speed)) if speed_col=="mach" else (pd.to_numeric(filtered["kias"], errors="coerce") == int(sel_speed)))
                )] if sel_speed is not None else filtered[filtered["cruise_alt"] == sel_alt]
                title_suffix = (f", {speed_label} {sel_speed}" if sel_speed is not None else "")
            else:
                # Optimization modes: ignore speed facet
                df_pr = filtered[filtered["cruise_alt"] == sel_alt]
                title_suffix = f", Optimized for {cruise_mode}"
            if len(df_pr) > 0:
                df_pr = df_pr.copy()
                df_pr["isa_label"] = df_pr["isa_dev"].apply(lambda x: f"ISA {x:+d}°C")
                fig = go.Figure()
                mods = sorted(df_pr["mod"].dropna().unique().tolist())
                aircrafts = sorted(df_pr["aircraft"].dropna().unique().tolist())
                color_map = {"Flatwing": "red", "Tamarack": "green"}
                for mod in mods:
                    for a in aircrafts:
                        for dev in sorted(df_pr["isa_dev"].dropna().unique()):
                            dfg0 = df_pr[(df_pr["mod"] == mod) & (df_pr["aircraft"] == a) & (df_pr["isa_dev"] == dev)].copy()
                            if dfg0.empty:
                                continue
                            # Legend metrics
                            att_m_vals = pd.to_numeric(dfg0.get("achieved_mach"), errors="coerce").dropna()
                            att_m_txt = f"M {att_m_vals.max():.2f}" if len(att_m_vals) > 0 else "M —"
                            att_alt_vals = pd.to_numeric(dfg0.get("first_level_off_ft"), errors="coerce").dropna()
                            att_fl_txt = f"FL{int(att_alt_vals.max()/100)}" if len(att_alt_vals) > 0 else "FL—"
                            # Prep series: choose x metric based on cruise mode
                            if cruise_mode == "Max Endurance":
                                dfg0.loc[:, "x_num"] = pd.to_numeric(dfg0["total_time_min"], errors="coerce")
                                x_axis_label = "Endurance (min)"
                                title_base = "Payload-Endurance"
                            else:
                                dfg0.loc[:, "x_num"] = pd.to_numeric(dfg0["total_dist_nm"], errors="coerce")
                                x_axis_label = "Range (NM)"
                                title_base = "Payload-Range"
                            dfg = dfg0.dropna(subset=["x_num", "payload"]).copy()
                            if dfg.empty:
                                continue
                            dfg = dfg.groupby("payload", as_index=False).agg(x_num=("x_num", "max"))
                            dfg = dfg.sort_values(["payload"], ascending=False).copy()
                            x_vals = dfg["x_num"].to_numpy()
                            y_vals = dfg["payload"].to_numpy()
                            x_mon = np.maximum.accumulate(x_vals)
                            x_plot = np.concatenate(([0.0], x_mon[:-1])) if len(x_mon) >= 1 else x_mon
                            if len(x_plot) > 1:
                                keep = np.concatenate(([True], x_plot[1:] > x_plot[:-1] + 1e-9))
                                x_plot = x_plot[keep]
                                y_vals = y_vals[keep]
                            label_mod = ("BL" if mod == "Flatwing" else "Mod")
                            fig.add_trace(go.Scatter(
                                x=x_plot,
                                y=y_vals,
                                mode="lines",
                                name=f"{label_mod} Max Achieved {att_m_txt}, Attained {att_fl_txt} - {a} - ISA {int(dev):+d}°C",
                                line=dict(color=color_map.get(mod, None))
                            ))
                fig.update_layout(
                    title=f"{title_base} at FL{int(sel_alt/100):.0f}{title_suffix}",
                    xaxis_title=x_axis_label,
                    yaxis_title="Payload (lb)",
                    template="plotly_white",
                    width=1000,
                    height=600,
                    xaxis=dict(showgrid=True, gridcolor="LightGrey", gridwidth=1)
                )
                st.plotly_chart(fig, use_container_width=False)
            else:
                st.info("No data for selected altitude.")

    # Max Range vs Mach (payload = 0)
    if "Max Range vs Mach" in plot_choices:
        st.subheader("Max Range vs Mach (Payload = 0)")
        # Choose altitude
        alt_options = sorted(filtered["cruise_alt"].unique().tolist(), reverse=True)
        if alt_options:
            sel_alt = st.selectbox("Altitude for Range vs Mach", options=alt_options, key="rvsm_alt")
            df_mach = filtered[(filtered["payload"] == 0) & (filtered["cruise_alt"] == sel_alt)]
            if len(df_mach) > 0:
                # Create ISA deviation labels for better legend
                df_mach = df_mach.copy()
                df_mach.loc[:, "isa_label"] = df_mach["isa_dev"].apply(lambda x: f"ISA {x:+d}°C")
                has_kias = ("kias" in df_mach.columns) and pd.to_numeric(df_mach.get("kias"), errors="coerce").notna().any()
                speed_col = "kias" if has_kias else "mach"
                x_label = "IAS (kts)" if speed_col == "kias" else "Target Mach"
                title_txt = f"Max Range vs {'IAS' if speed_col=='kias' else 'Mach'} at FL{int(sel_alt/100):.0f} Goal"
                # Build custom legend: show attained first level-off altitude
                fig = go.Figure()
                color_map = {"Flatwing": "red", "Tamarack": "green"}
                dash_for = { -20: "longdashdot", -10: "dash", 0: "solid", 10: "dot", 20: "dashdot", 30: "longdash" }
                for mod in sorted(df_mach["mod"].dropna().unique().tolist()):
                    d_mod = df_mach[df_mach["mod"] == mod]
                    for dev in sorted(pd.to_numeric(d_mod["isa_dev"], errors="coerce").dropna().unique().tolist()):
                        d_dev = d_mod[pd.to_numeric(d_mod["isa_dev"], errors="coerce") == dev]
                        # Compute attained first level-off altitude (max across points)
                        att_alt = pd.to_numeric(d_dev.get("first_level_off_ft"), errors="coerce").dropna()
                        att_fl_txt = f"FL{int(att_alt.max()/100)}" if len(att_alt) > 0 else "FL—"
                        # Plot one trace per aircraft to preserve shapes; only first shows legend
                        first_legend = True
                        for ac in sorted(d_dev["aircraft"].dropna().unique().tolist()):
                            d_line = d_dev[d_dev["aircraft"] == ac]
                            label_mod = ("BL" if mod == "Flatwing" else "Mod")
                            fig.add_trace(go.Scatter(
                                x=d_line[speed_col],
                                y=d_line["total_dist_nm"],
                                mode="lines",
                                name=(f"{label_mod} Attained {att_fl_txt} - ISA {int(dev):+d}°C" if first_legend else None),
                                line=dict(color=color_map.get(mod, None), dash=dash_for.get(int(dev), "solid")),
                                showlegend=first_legend
                            ))
                            first_legend = False
                fig.update_layout(title=title_txt, xaxis_title=x_label, yaxis_title="Max Range (NM)", template="plotly_white", width=1000, height=600, xaxis=dict(showgrid=True, gridcolor="LightGrey", gridwidth=1))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No payload=0 data at selected altitude.")

    # Max Endurance vs Mach (payload = 0)
    if "Max Endurance vs Mach" in plot_choices:
        st.subheader("Max Endurance vs Mach (Payload = 0)")
        alt_options = sorted(filtered["cruise_alt"].unique().tolist(), reverse=True)
        if alt_options:
            sel_alt = st.selectbox("Altitude for Endurance vs Mach", options=alt_options, key="evsm_alt")
            df_end = filtered[(filtered["payload"] == 0) & (filtered["cruise_alt"] == sel_alt)]
            if len(df_end) > 0:
                # Create ISA deviation labels for better legend
                df_end = df_end.copy()
                df_end.loc[:, "isa_label"] = df_end["isa_dev"].apply(lambda x: f"ISA {x:+d}°C")
                has_kias = ("kias" in df_end.columns) and pd.to_numeric(df_end.get("kias"), errors="coerce").notna().any()
                speed_col = "kias" if has_kias else "mach"
                x_label = "IAS (kts)" if speed_col == "kias" else "Target Mach"
                title_txt = f"Max Endurance vs {'IAS' if speed_col=='kias' else 'Mach'} at FL{int(sel_alt/100):.0f} Goal"
                # Build custom legend: show attained first level-off altitude
                fig = go.Figure()
                color_map = {"Flatwing": "red", "Tamarack": "green"}
                dash_for = { -20: "longdashdot", -10: "dash", 0: "solid", 10: "dot", 20: "dashdot", 30: "longdash" }
                for mod in sorted(df_end["mod"].dropna().unique().tolist()):
                    d_mod = df_end[df_end["mod"] == mod]
                    for dev in sorted(pd.to_numeric(d_mod["isa_dev"], errors="coerce").dropna().unique().tolist()):
                        d_dev = d_mod[pd.to_numeric(d_mod["isa_dev"], errors="coerce") == dev]
                        # Attained first level-off altitude
                        att_alt = pd.to_numeric(d_dev.get("first_level_off_ft"), errors="coerce").dropna()
                        att_fl_txt = f"FL{int(att_alt.max()/100)}" if len(att_alt) > 0 else "FL—"
                        first_legend = True
                        for ac in sorted(d_dev["aircraft"].dropna().unique().tolist()):
                            d_line = d_dev[d_dev["aircraft"] == ac]
                            label_mod = ("BL" if mod == "Flatwing" else "Mod")
                            fig.add_trace(go.Scatter(
                                x=d_line[speed_col],
                                y=d_line["total_time_min"],
                                mode="lines",
                                name=(f"{label_mod} Attained {att_fl_txt} - ISA {int(dev):+d}°C" if first_legend else None),
                                line=dict(color=color_map.get(mod, None), dash=dash_for.get(int(dev), "solid")),
                                showlegend=first_legend
                            ))
                            first_legend = False
                fig.update_layout(title=title_txt, xaxis_title=x_label, yaxis_title="Max Endurance (min)", template="plotly_white", xaxis=dict(showgrid=True, gridcolor="LightGrey", gridwidth=1))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No payload=0 data at selected altitude.")

    # Max Range vs Altitude (payload = 0)
    if "Max Range vs Altitude" in plot_choices:
        # Dynamic subheader: if optimizing endurance, label accordingly
        if cruise_mode == "Max Endurance":
            st.subheader("Max Endurance vs Altitude (Payload = 0)")
        else:
            st.subheader("Max Range vs Altitude (Payload = 0)")
        # Choose speed (Mach for jets, IAS for turboprops)
        has_kias = ("kias" in filtered.columns) and pd.to_numeric(filtered.get("kias"), errors="coerce").notna().any()
        speed_col = "kias" if has_kias else "mach"
        speed_label = "IAS (kts)" if speed_col == "kias" else "Mach"
        if has_kias:
            speed_options = sorted(pd.to_numeric(filtered["kias"], errors="coerce").dropna().unique().tolist(), reverse=True)
        else:
            speed_options = sorted(pd.to_numeric(filtered["mach"], errors="coerce").dropna().unique().tolist(), reverse=True)
        if speed_options:
            sel_speed = st.selectbox(f"{speed_label} for Range vs Altitude", options=speed_options, key="rva_speed")
            if speed_col == "mach":
                df_alt = filtered[(filtered["payload"] == 0) & (np.isclose(pd.to_numeric(filtered["mach"], errors="coerce"), float(sel_speed)))]
            else:
                df_alt = filtered[(filtered["payload"] == 0) & (pd.to_numeric(filtered["kias"], errors="coerce") == int(sel_speed))]
            if cruise_mode == "MCT (Max Thrust)":
                title_txt = (f"Max Range vs {'IAS' if speed_col=='kias' else 'Mach'} at M {float(sel_speed):.2f} Goal" if speed_col == "mach" else f"Max Range vs {'IAS' if speed_col=='kias' else 'Mach'} at IAS {int(sel_speed)} kt")
            else:
                if cruise_mode == "Max Endurance":
                    title_txt = "Max Endurance vs Altitude (Optimized)"
                else:
                    title_txt = "Max Range vs Altitude (Optimized)"
        else:
            df_alt = filtered[filtered["payload"] == 0]
            if cruise_mode == "MCT (Max Thrust)":
                title_txt = "Max Range vs Altitude"
            else:
                if cruise_mode == "Max Endurance":
                    title_txt = "Max Endurance vs Altitude (Optimized)"
                else:
                    title_txt = "Max Range vs Altitude (Optimized)"
        if len(df_alt) > 0:
            # Create ISA deviation labels for better legend
            df_alt = df_alt.copy()
            df_alt.loc[:, "isa_label"] = df_alt["isa_dev"].apply(lambda x: f"ISA {x:+d}°C")
            # Build custom legend: show attained Mach and attained first level-off altitude
            fig = go.Figure()
            color_map = {"Flatwing": "red", "Tamarack": "green"}
            dash_for = { -20: "longdashdot", -10: "dash", 0: "solid", 10: "dot", 20: "dashdot", 30: "longdash" }
            for mod in sorted(df_alt["mod"].dropna().unique().tolist()):
                d_mod = df_alt[df_alt["mod"] == mod]
                for dev in sorted(pd.to_numeric(d_mod["isa_dev"], errors="coerce").dropna().unique().tolist()):
                    d_dev = d_mod[pd.to_numeric(d_mod["isa_dev"], errors="coerce") == dev]
                    # Compute attained metrics
                    att_m = pd.to_numeric(d_dev.get("achieved_mach"), errors="coerce").dropna()
                    att_m_txt = f"M {att_m.max():.2f}" if len(att_m) > 0 else "M —"
                    att_alt = pd.to_numeric(d_dev.get("first_level_off_ft"), errors="coerce").dropna()
                    att_fl_txt = f"FL{int(att_alt.max()/100)}" if len(att_alt) > 0 else "FL—"
                    first_legend = True
                    for ac in sorted(d_dev["aircraft"].dropna().unique().tolist()):
                        d_line = d_dev[d_dev["aircraft"] == ac]
                        label_mod = ("BL" if mod == "Flatwing" else "Mod")
                        # Choose Y metric based on mode: Range (NM) or Endurance (min)
                        if cruise_mode == "Max Endurance":
                            y_vals = d_line["total_time_min"]
                            y_label = "Max Endurance (min)"
                        else:
                            y_vals = d_line["total_dist_nm"]
                            y_label = "Max Range (NM)"
                        fig.add_trace(go.Scatter(
                            x=d_line["cruise_alt"],
                            y=y_vals,
                            mode="lines",
                            name=(f"{label_mod} Max Achieved {att_m_txt}, Attained {att_fl_txt} - ISA {int(dev):+d}°C" if first_legend else None),
                            line=dict(color=color_map.get(mod, None), dash=dash_for.get(int(dev), "solid")),
                            showlegend=first_legend
                        ))
                        first_legend = False
            fig.update_layout(title=title_txt, xaxis_title="Target Altitude (ft)", yaxis_title=y_label, template="plotly_white", xaxis=dict(showgrid=True, gridcolor="LightGrey", gridwidth=1))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No payload=0 data at selected Mach.")

    # Endurance vs Altitude (payload = 0)
    if "Endurance vs Altitude" in plot_choices:
        st.subheader("Endurance vs Altitude (Payload = 0)")
        if cruise_mode == "MCT (Max Thrust)":
            # Choose speed (Mach for jets, IAS for turboprops)
            has_kias = ("kias" in filtered.columns) and pd.to_numeric(filtered.get("kias"), errors="coerce").notna().any()
            speed_col = "kias" if has_kias else "mach"
            speed_label = "IAS (kts)" if speed_col == "kias" else "Mach"
            speed_options = sorted(pd.to_numeric(filtered[speed_col], errors="coerce").dropna().unique().tolist(), reverse=True)
            if speed_options:
                sel_speed = st.selectbox(f"{speed_label} for Endurance vs Altitude", options=speed_options, key="eva_speed")
                if speed_col == "mach":
                    df_alt = filtered[(filtered["payload"] == 0) & (np.isclose(pd.to_numeric(filtered["mach"], errors="coerce"), float(sel_speed)))]
                else:
                    df_alt = filtered[(filtered["payload"] == 0) & (pd.to_numeric(filtered["kias"], errors="coerce") == int(sel_speed))]
            else:
                df_alt = filtered[filtered["payload"] == 0]
            title_txt = (f"Max Endurance vs {'IAS' if speed_col=='kias' else 'Mach'} at selected speed")
        else:
            # Optimization modes: ignore speed facet
            df_alt = filtered[filtered["payload"] == 0]
            title_txt = "Max Endurance vs Altitude (Optimized)"
        if len(df_alt) > 0:
            # Create ISA deviation labels for better legend
            df_alt = df_alt.copy()
            df_alt.loc[:, "isa_label"] = df_alt["isa_dev"].apply(lambda x: f"ISA {x:+d}°C")
            fig = go.Figure()
            color_map = {"Flatwing": "red", "Tamarack": "green"}
            dash_for = { -20: "longdashdot", -10: "dash", 0: "solid", 10: "dot", 20: "dashdot", 30: "longdash" }
            for mod in sorted(df_alt["mod"].dropna().unique().tolist()):
                d_mod = df_alt[df_alt["mod"] == mod]
                for dev in sorted(pd.to_numeric(d_mod["isa_dev"], errors="coerce").dropna().unique().tolist()):
                    d_dev = d_mod[pd.to_numeric(d_mod["isa_dev"], errors="coerce") == dev]
                    # Attained first level-off altitude
                    att_alt = pd.to_numeric(d_dev.get("first_level_off_ft"), errors="coerce").dropna()
                    att_fl_txt = f"FL{int(att_alt.max()/100)}" if len(att_alt) > 0 else "FL—"
                    first_legend = True
                    for ac in sorted(d_dev["aircraft"].dropna().unique().tolist()):
                        d_line = d_dev[d_dev["aircraft"] == ac]
                        label_mod = ("BL" if mod == "Flatwing" else "Mod")
                        fig.add_trace(go.Scatter(
                            x=d_line["cruise_alt"],
                            y=d_line["total_time_min"],
                            mode="lines",
                            name=(f"{label_mod} Attained {att_fl_txt} - ISA {int(dev):+d}°C" if first_legend else None),
                            line=dict(color=color_map.get(mod, None), dash=dash_for.get(int(dev), "solid")),
                            showlegend=first_legend
                        ))
                        first_legend = False
            fig.update_layout(title=title_txt, xaxis_title="Target Altitude (ft)", yaxis_title="Max Endurance (min)", template="plotly_white", xaxis=dict(showgrid=True, gridcolor="LightGrey", gridwidth=1))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No payload=0 data for endurance vs altitude.")

else:
    st.info("Configure parameters in the sidebar and click 'Run Sweep.'")
