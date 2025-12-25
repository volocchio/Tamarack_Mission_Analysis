import os
import runpy
import io
import zipfile
from datetime import datetime
import streamlit as st
import pandas as pd
from aircraft_config import AIRCRAFT_CONFIG
from utils import load_airports
from simulation import run_simulation as _run_simulation, haversine_with_bearing
from display import display_simulation_results


def run_simulation(*args, **kwargs):
    try:
        return _run_simulation(*args, **kwargs)
    except Exception as e:
        try:
            st.error(f"Simulation failed: {e}")
        except Exception:
            pass
        return pd.DataFrame(), {'error': str(e)}, 0.0, 0.0, 0.0, 0.0, ""

# --- Streamlit UI ---
st.set_page_config(page_title="Flight Simulation App", layout="wide")
mode = st.radio("Mode", ["Single Run", "Batch Sweeps"], index=0, horizontal=True, key="mode_selector")
if mode == "Batch Sweeps":
    batch_path = os.path.join(os.path.dirname(__file__), "batch_app.py")
    _orig_spc = st.set_page_config
    try:
        st.set_page_config = lambda *args, **kwargs: None
        runpy.run_path(batch_path, run_name="__main__")
    finally:
        st.set_page_config = _orig_spc
    st.stop()
st.title("Flight Simulation App")
st.markdown("""
This app simulates a flight between two airports using a specified aircraft model.
It calculates flight parameters such as altitude, speed, thrust, and drag over time,
and visualizes the flight profile with charts.
""")

main = st.container()

# Load airports data
airports_df = load_airports()

# Ensure we have the required columns
if 'display_name' not in airports_df.columns or 'ident' not in airports_df.columns:
    st.error("Error: Required columns not found in airports data")
    st.stop()

# Create a mapping of upper case display names to original display names
display_name_mapping = {name.upper(): name for name in airports_df["display_name"].unique()}

# Get the list of display names for the selectbox (using original case)
airport_display_names = list(display_name_mapping.values())

# Create a reverse mapping from display name to airport info
display_name_to_info = {}
for _, row in airports_df.iterrows():
    display_name = row["display_name"]
    if display_name not in display_name_to_info:
        display_name_to_info[display_name] = row

# Initialize session state
if 'initial_values' not in st.session_state:
    st.session_state.initial_values = {}
if 'last_weight_option' not in st.session_state:
    st.session_state.last_weight_option = None

# Sidebar inputs
with st.sidebar:
    st.header("Flight Parameters")
    
    # Aircraft model selection
    aircraft_types = sorted({a for (a, _) in AIRCRAFT_CONFIG.keys()})
    aircraft_model = st.selectbox("Aircraft Model", aircraft_types, index=aircraft_types.index("CJ1") if "CJ1" in aircraft_types else 0, key="aircraft_model")
    
    # Update BOW and other values when aircraft changes
    if 'last_aircraft' not in st.session_state or st.session_state.last_aircraft != aircraft_model:
        st.session_state.last_aircraft = aircraft_model
        # Clear Tamarack BOW to force update
        if 'bow_tamarack' in st.session_state:
            del st.session_state.bow_tamarack
        # Clear payload inputs to reset them
        if 'payload_input_flatwing' in st.session_state:
            del st.session_state.payload_input_flatwing
        if 'payload_input_tamarack' in st.session_state:
            del st.session_state.payload_input_tamarack

    # Display both aircraft images
    if aircraft_model:
        try:
            image_path = f"images/tamarack_{aircraft_model}.jpg"
            st.image(image_path, caption=f"Tamarack {aircraft_model}", use_container_width=True)
        except Exception:
            st.warning(f"Tamarack image not found: {image_path}")
        try:
            image_path = f"images/flatwing_{aircraft_model}.jpg"
            st.image(image_path, caption=f"Flatwing {aircraft_model}", use_container_width=True)
        except Exception:
            st.warning(f"Flatwing image not found: {image_path}")

    # Load aircraft config first
    mods_available = [m for (a, m) in AIRCRAFT_CONFIG if a == aircraft_model]
    if not mods_available:
        st.error(f"No modifications available for aircraft model {aircraft_model}.")
        st.stop()

    # Get default configuration (Flatwing)
    flatwing_config = AIRCRAFT_CONFIG.get((aircraft_model, "Flatwing"))
    if not flatwing_config:
        st.error(f"No Flatwing configuration found for {aircraft_model}.")
        st.stop()

    # Get Tamarack configuration if available
    tamarack_config = AIRCRAFT_CONFIG.get((aircraft_model, "Tamarack"))
    
    # Extract Flatwing configuration values
    try:
        config_values = list(flatwing_config)[:35]
        s, b, e, h, sweep_25c, sfc, engines_orig, thrust_mult, ceiling, CL0, CLA, cdo, dcdo_flap1, dcdo_flap2, \
            dcdo_flap3, dcdo_gear, mu_to, mu_lnd, bow, mzfw, mrw, mtow, max_fuel, \
            taxi_fuel_default, reserve_fuel_default, mmo, VMO, clmax, clmax_1, clmax_2, m_climb, \
            v_climb, roc_min, m_descent, v_descent = config_values
            
        # Store MZFW and BOW for Flatwing
        flatwing_mzfw = mzfw
        flatwing_bow = bow
        
        # Get Tamarack MZFW and BOW if available
        if tamarack_config:
            tamarack_values = list(tamarack_config)[:35]
            tamarack_mzfw = tamarack_values[19]  # MZFW is at index 19
            tamarack_bow = tamarack_values[18]   # BOW is at index 18
            tamarack_mrw = tamarack_values[20]
            tamarack_mtow = tamarack_values[21]
            tamarack_max_fuel = tamarack_values[22]
            tamarack_taxi_fuel_default = tamarack_values[23]
            tamarack_reserve_fuel_default = tamarack_values[24]
        else:
            tamarack_mzfw = mzfw
            tamarack_bow = bow
            tamarack_mrw = mrw
            tamarack_mtow = mtow
            tamarack_max_fuel = max_fuel
            tamarack_taxi_fuel_default = taxi_fuel_default
            tamarack_reserve_fuel_default = reserve_fuel_default
            
    except ValueError as e:
        st.error(f"Error extracting configuration values: {str(e)}")
        st.stop()

    # Airport selection
    st.subheader('Flight Plan')
    departure_airport = st.selectbox(
        "Departure Airport",
        options=airport_display_names,
        index=next((i for i, name in enumerate(airport_display_names) 
                  if name.startswith("KSZT")), 0),
        format_func=lambda x: x,
        key="departure_airport"
    )
    
    arrival_airport = st.selectbox(
        "Arrival Airport",
        options=airport_display_names,
        index=next((i for i, name in enumerate(airport_display_names) 
                  if name.startswith("KSJT")), 0),
        format_func=lambda x: x,
        key="arrival_airport"
    )

    try:
        dep_airport_code = str(display_name_to_info[departure_airport]["ident"]).upper()
        arr_airport_code = str(display_name_to_info[arrival_airport]["ident"]).upper()
    except Exception:
        dep_airport_code = str(departure_airport).split(" - ")[0].upper()
        arr_airport_code = str(arrival_airport).split(" - ")[0].upper()

    # Weight mode selection
    weight_option = st.radio("Weight Configuration", [
        "Manual Input",
        "Max Fuel (Fill Tanks, Adjust Payload to MRW)",
        "Max Payload (Fill Payload to MZFW, Adjust Fuel to MRW)"
    ], index=0, key="weight_option")

    # Track if weight option changed
    weight_option_changed = st.session_state.get('last_weight_option') != st.session_state.weight_option
    st.session_state.last_weight_option = st.session_state.weight_option

    # Get initial values based on weight option
    if weight_option == "Max Fuel (Fill Tanks, Adjust Payload to MRW)":
        initial_fuel = float(max_fuel)
        initial_payload = float(min(flatwing_mzfw - flatwing_bow, mrw - (flatwing_bow + max_fuel)))
        rw = flatwing_bow + initial_payload + initial_fuel
        tow = rw - taxi_fuel_default
        if tow > mtow:
            initial_fuel = mtow - (flatwing_bow + initial_payload) + taxi_fuel_default
            if initial_fuel < 0:
                initial_fuel = 0
                initial_payload = mtow - flatwing_bow + taxi_fuel_default
    elif weight_option == "Max Payload (Fill Payload to MZFW, Adjust Fuel to MRW)":
        initial_payload = float(flatwing_mzfw - flatwing_bow)  # Calculate max payload as MZFW - BOW
        initial_fuel = float(min(max_fuel, mrw - (flatwing_bow + initial_payload)))
        if initial_fuel == max_fuel:
            initial_payload = float(min(flatwing_mzfw - flatwing_bow, mrw - (flatwing_bow + max_fuel)))
        
        # Update Tamarack payload in session state when Max Payload is selected
        if 'bow_tamarack' in st.session_state:
            tamarack_max_payload = max(0, tamarack_mzfw - st.session_state.bow_tamarack)
    else:
# Set initial values based on aircraft model
        if aircraft_model == "M2":
            initial_fuel = 3440.0  # Default fuel for M2
        else:
            initial_fuel = 3440.0  # Default fuel for other models
        initial_payload = 0.0

    # Prevent negative payloads
    initial_payload = max(0, initial_payload)
    initial_fuel = max(0, initial_fuel)
    initial_fuel = min(initial_fuel, max_fuel)

    # Store initial values in session state
    st.session_state.initial_values = {
        'payload': int(initial_payload),
        'fuel': int(initial_fuel),
        'taxi_fuel': int(taxi_fuel_default),
        'reserve_fuel': int(reserve_fuel_default),
        'cruise_altitude': int(ceiling)
    }

    # Weight inputs - Flatwing
    st.subheader('Flatwing Weight Adjustment')
    
    # Create three columns for the inputs
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # First column - BOW and Payload
        bow_f = st.number_input(
            "BOW (lb)",
            min_value=0,
            max_value=int(flatwing_mzfw),
            value=int(flatwing_bow),
            step=100,
            help="Basic Operating Weight (Empty Weight + pilot)",
            key="bow_input_flatwing",
            on_change=lambda: st.session_state.update({"bow_changed": True})
        )
        
        # Calculate max payload based on BOW and MZFW
        max_payload_f = max(0, flatwing_mzfw - bow_f)
        
        # Calculate the payload value before creating the widget
        payload_value = 0
        # Always ensure the payload doesn't exceed max_payload_f
        current_payload = st.session_state.get('payload_input_flatwing', 0)
        payload_value = min(current_payload, int(max_payload_f))
        
        # Update payload if we're in Max Payload mode and either:
        # 1. The weight option just changed to Max Payload, or
        # 2. BOW was just modified while in Max Payload mode
        if weight_option == "Max Payload (Fill Payload to MZFW, Adjust Fuel to MRW)" and \
           (weight_option_changed or st.session_state.get("bow_changed", False)):
            payload_value = int(max_payload_f)
        
        # Reset the flag after processing
        st.session_state["bow_changed"] = False
        
        # Ensure the payload input never exceeds max_payload_f
        payload_input_f = st.number_input(
            "Payload (lb)",
            min_value=0,
            max_value=int(max_payload_f),
            value=payload_value,
            step=100,
            help=f"Maximum payload: {int(max_payload_f):,} lb (MZFW: {int(flatwing_mzfw):,} - BOW)",
            key="payload_input_flatwing",
            on_change=lambda: st.session_state.update({"payload_changed_f": True})
        )
    
    with col2:
        # Second column - Fuel inputs
        fuel_input_f = st.number_input(
            "Fuel (lb)",
            min_value=0,
            max_value=int(max_fuel),
            value=st.session_state.initial_values.get('fuel', int(initial_fuel)),
            step=100,
            help=f"Maximum fuel: {int(max_fuel):,} lb",
            key="fuel_input_flatwing"
        )
        
        reserve_fuel_f = st.number_input(
            "Reserve Fuel (lb)",
            min_value=0,
            value=st.session_state.initial_values.get('reserve_fuel', int(reserve_fuel_default)),
            step=10,
            key="reserve_fuel_input_flatwing"
        )
    
    with col3:
        # Third column - Taxi and Altitude
        taxi_fuel_f = st.number_input(
            "Taxi Fuel (lb)",
            min_value=0,
            value=st.session_state.initial_values.get('taxi_fuel', int(taxi_fuel_default)),
            step=10,
            key="taxi_fuel_input_flatwing"
        )
        
        cruise_altitude_f = st.number_input(
            "Cruise Altitude Goal (ft)",
            min_value=0,
            max_value=int(ceiling),
            value=st.session_state.initial_values.get('cruise_altitude', int(ceiling)),
            step=1000,
            key="cruise_altitude_input_flatwing"
        )

    # Weight inputs - Tamarack
    st.subheader('Tamarack Weight Adjustment')
    
    # Determine if we're in comparison mode (both Flatwing and Tamarack are shown)
    comparison_mode = tamarack_config is not None
    
    # Initialize Tamarack BOW if not set
    if 'bow_tamarack' not in st.session_state:
        if comparison_mode:
            # In comparison mode, set Tamarack BOW based on Flatwing BOW
            bow_diff = 65 if aircraft_model == "M2" else 75
            st.session_state.bow_tamarack = st.session_state.get('bow_flatwing', int(flatwing_bow)) + bow_diff
        else:
            # In single config mode, use the configured Tamarack BOW
            st.session_state.bow_tamarack = int(tamarack_bow)
    
    # Update Tamarack BOW if Flatwing BOW changes in comparison mode
    if comparison_mode and st.session_state.get('bow_changed', False):
        bow_diff = 65 if aircraft_model == "M2" else 75
        st.session_state.bow_tamarack = st.session_state.bow_flatwing + bow_diff
    
    # Create three columns for the inputs
    col4, col5, col6 = st.columns(3)
    
    with col4:
        # BOW input for Tamarack
        bow_t = st.number_input(
            "BOW (lb)",
            min_value=0,
            max_value=int(tamarack_mzfw),
            value=st.session_state.bow_tamarack,
            step=100,
            help="Basic Operating Weight (Empty Weight + pilot)" + 
                 (" - 75 lbs heavier than Flatwing (65 lbs for M2)" if comparison_mode else ""),
            key="bow_input_tamarack",
            disabled=comparison_mode,  # Disable in comparison mode
            on_change=lambda: st.session_state.update({"bow_changed_t": True})
        )
        
        # Store the BOW in session state
        st.session_state.bow_tamarack = bow_t
        
        # Calculate max payload based on current BOW and Tamarack MZFW
        max_payload_t = max(0, tamarack_mzfw - bow_t)
        
        # Calculate the payload value before creating the widget
        payload_value = 0
        # Always ensure the payload doesn't exceed max_payload_t
        current_payload = st.session_state.get('payload_input_tamarack', 0)
        payload_value = min(current_payload, int(max_payload_t))
        
        # Update payload if we're in Max Payload mode and either:
        # 1. The weight option just changed to Max Payload, or
        # 2. BOW was just modified while in Max Payload mode
        if weight_option == "Max Payload (Fill Payload to MZFW, Adjust Fuel to MRW)" and \
           (weight_option_changed or st.session_state.get("bow_changed_t", False)):
            payload_value = int(max_payload_t)
        
        # Reset the flag after processing
        st.session_state["bow_changed_t"] = False
        
        # Ensure the payload input never exceeds max_payload_t
        payload_input_t = st.number_input(
            "Payload (lb)",
            min_value=0,
            max_value=int(max_payload_t),
            value=payload_value,
            step=100,
            help=f"Maximum payload: {int(max_payload_t):,} lb (MZFW: {int(tamarack_mzfw):,} - BOW)",
            key="payload_input_tamarack",
            on_change=lambda: st.session_state.update({"payload_changed_t": True})
        )
    
    with col5:
        # Second column - Fuel inputs
        fuel_input_t = st.number_input(
            "Fuel (lb)",
            min_value=0,
            max_value=int(max_fuel),
            value=st.session_state.initial_values.get('fuel', int(initial_fuel)),
            step=100,
            help=f"Maximum fuel: {int(max_fuel):,} lb",
            key="fuel_input_tamarack"
        )
        
        reserve_fuel_t = st.number_input(
            "Reserve Fuel (lb)",
            min_value=0,
            value=st.session_state.initial_values.get('reserve_fuel', int(reserve_fuel_default)),
            step=10,
            key="reserve_fuel_input_tamarack"
        )
    
    with col6:
        # Third column - Taxi and Altitude
        taxi_fuel_t = st.number_input(
            "Taxi Fuel (lb)",
            min_value=0,
            value=st.session_state.initial_values.get('taxi_fuel', int(taxi_fuel_default)),
            step=10,
            key="taxi_fuel_input_tamarack"
        )
        
        cruise_altitude_t = st.number_input(
            "Cruise Altitude Goal (ft)",
            min_value=0,
            max_value=int(ceiling),
            value=st.session_state.initial_values.get('cruise_altitude', int(ceiling)),
            step=1000,
            key="cruise_altitude_input_tamarack"
        )

    payload_f = payload_input_f
    fuel_f = fuel_input_f
    try:
        payload_t = payload_input_t
        fuel_t = fuel_input_t
    except Exception:
        payload_t = 0
        fuel_t = 0

    # Wing type selection
    st.subheader('Aircraft Configuration')
    wing_type = st.radio("Wing Type", ["Flatwing", "Tamarack", "Comparison"], index=2, key="wing_type")
    if wing_type != "Comparison" and wing_type not in mods_available:
        st.error(f"Wing type '{wing_type}' is not available for aircraft model {aircraft_model}. Available options: {mods_available}")
        st.stop()

    # Takeoff flap selection
    flap_option = st.radio("Takeoff Flaps", ["Flap 0", "Flaps 15"], index=0)
    takeoff_flap = 1 if flap_option == "Flaps 15" else 0

    # Winds and temps source
    try:
        if st.session_state.get("winds_temps_source") in ("Current Conditions", "Monthly Climatology"):
            st.session_state["winds_temps_source"] = "No Wind"
    except Exception:
        pass
    winds_temps_source = st.radio(
        "Winds and Temps Aloft Source",
        ["No Wind", "Summer Average", "Winter Average"],
        index=0,
        key="winds_temps_source",
    )

    # ISA deviation
    isa_dev = int(st.number_input("ISA Deviation (C)", value=0.0, step=1.0))

    # Performance tuning biases
    st.subheader("Performance Tuning (Biases)")
    st.markdown("Adjust SFC and thrust vs altitude to better match specific aircraft behavior.")
    bias_mode = st.radio(
        "Bias Mode",
        ["Manual", "Auto"],
        index=0,
        key="bias_mode",
        help="Manual: set tri-level per-mod biases. Auto: enter real-world data per mod."
    )

    def _ss_int(key: str, default: int = 0) -> int:
        try:
            return int(st.session_state.get(key, default))
        except Exception:
            return int(default)

    def _baseline_cdo_k(mod: str) -> tuple[float, float]:
        try:
            ac = AIRCRAFT_CONFIG.get((aircraft_model, mod))
            if not ac:
                return 0.0, 0.0
            s = float(ac[0])
            b = float(ac[1])
            e = float(ac[2])
            h = float(ac[3])
            cdo = float(ac[11])
            a = b ** 2 / s * (1 + 1.9 * h / b)
            k = 1 / (3.14159 * e * a)
            return cdo, float(k)
        except Exception:
            return 0.0, 0.0

    for _k in (
        'tam_sfc_low', 'tam_sfc_mid', 'tam_sfc_high', 'tam_thrust_low', 'tam_thrust_mid', 'tam_thrust_high',
        'flat_sfc_low', 'flat_sfc_mid', 'flat_sfc_high', 'flat_thrust_low', 'flat_thrust_mid', 'flat_thrust_high',
        'tam_drag_cdo_delta_pct', 'tam_drag_k_delta_pct', 'flat_drag_cdo_delta_pct', 'flat_drag_k_delta_pct',
        'autobias_target',
    ):
        try:
            if _k not in st.session_state:
                st.session_state[_k] = ("Both" if _k == 'autobias_target' else 0)
        except Exception:
            pass

    try:
        _prev_bias_mode = st.session_state.get('_prev_bias_mode')
    except Exception:
        _prev_bias_mode = None
    if _prev_bias_mode != bias_mode:
        st.session_state['_prev_bias_mode'] = bias_mode
        if bias_mode == "Manual":
            applied = st.session_state.get('applied_biases', {})
            tam_applied = applied.get('Tamarack')
            flat_applied = applied.get('Flatwing')
            if isinstance(tam_applied, dict):
                for k, ss_k in (
                    ('sfc_low', 'tam_sfc_low'),
                    ('sfc_mid', 'tam_sfc_mid'),
                    ('sfc_high', 'tam_sfc_high'),
                    ('thrust_low', 'tam_thrust_low'),
                    ('thrust_mid', 'tam_thrust_mid'),
                    ('thrust_high', 'tam_thrust_high'),
                    ('drag_cdo_delta_pct', 'tam_drag_cdo_delta_pct'),
                    ('drag_k_delta_pct', 'tam_drag_k_delta_pct'),
                ):
                    st.session_state[ss_k] = _ss_int(ss_k, _ss_int(ss_k, tam_applied.get(k, 0)))
                    try:
                        st.session_state[ss_k] = int(tam_applied.get(k, st.session_state[ss_k]))
                    except Exception:
                        pass
            if isinstance(flat_applied, dict):
                for k, ss_k in (
                    ('sfc_low', 'flat_sfc_low'),
                    ('sfc_mid', 'flat_sfc_mid'),
                    ('sfc_high', 'flat_sfc_high'),
                    ('thrust_low', 'flat_thrust_low'),
                    ('thrust_mid', 'flat_thrust_mid'),
                    ('thrust_high', 'flat_thrust_high'),
                    ('drag_cdo_delta_pct', 'flat_drag_cdo_delta_pct'),
                    ('drag_k_delta_pct', 'flat_drag_k_delta_pct'),
                ):
                    st.session_state[ss_k] = _ss_int(ss_k, _ss_int(ss_k, flat_applied.get(k, 0)))
                    try:
                        st.session_state[ss_k] = int(flat_applied.get(k, st.session_state[ss_k]))
                    except Exception:
                        pass
    bias_alt_mid_ft = st.number_input("Mid Altitude Breakpoint (ft)", min_value=0, max_value=int(ceiling), value=20000, step=1000)
    if bias_mode == "Manual":
        st.caption("Advanced: Per-mod Bias (tri-level)")
        col_mod1, col_mod2 = st.columns(2)
        with col_mod1:
            st.markdown("**Tamarack Biases**")
            tam_sfc_low = st.slider("Tamarack SFC Bias Low (%)", min_value=-20, max_value=20, step=1, key="tam_sfc_low")
            tam_sfc_mid = st.slider("Tamarack SFC Bias Mid (%)", min_value=-20, max_value=20, step=1, key="tam_sfc_mid")
            tam_sfc_high = st.slider("Tamarack SFC Bias High (%)", min_value=-20, max_value=20, step=1, key="tam_sfc_high")
            tam_thrust_low = st.slider("Tamarack Thrust Bias Low (%)", min_value=-20, max_value=20, step=1, key="tam_thrust_low")
            tam_thrust_mid = st.slider("Tamarack Thrust Bias Mid (%)", min_value=-20, max_value=20, step=1, key="tam_thrust_mid")
            tam_thrust_high = st.slider("Tamarack Thrust Bias High (%)", min_value=-20, max_value=20, step=1, key="tam_thrust_high")

            _tam_cdo_base, _tam_k_base = _baseline_cdo_k("Tamarack")
            _tam_cdo_line = st.empty()
            tam_drag_cdo_delta_pct = st.slider("Tamarack CDO Delta (%)", min_value=-10, max_value=10, step=1, key="tam_drag_cdo_delta_pct")
            try:
                _tam_cdo_line.caption(f"CDO: {_tam_cdo_base * (1.0 + float(tam_drag_cdo_delta_pct) / 100.0):.6f}  (base={_tam_cdo_base:.6f})")
            except Exception:
                pass
            _tam_k_line = st.empty()
            tam_drag_k_delta_pct = st.slider("Tamarack K Delta (%)", min_value=-10, max_value=10, step=1, key="tam_drag_k_delta_pct")
            try:
                _tam_k_line.caption(f"K: {_tam_k_base * (1.0 + float(tam_drag_k_delta_pct) / 100.0):.6f}  (base={_tam_k_base:.6f})")
            except Exception:
                pass
        with col_mod2:
            st.markdown("**Flatwing Biases**")
            flat_sfc_low = st.slider("Flatwing SFC Bias Low (%)", min_value=-20, max_value=20, step=1, key="flat_sfc_low")
            flat_sfc_mid = st.slider("Flatwing SFC Bias Mid (%)", min_value=-20, max_value=20, step=1, key="flat_sfc_mid")
            flat_sfc_high = st.slider("Flatwing SFC Bias High (%)", min_value=-20, max_value=20, step=1, key="flat_sfc_high")
            flat_thrust_low = st.slider("Flatwing Thrust Bias Low (%)", min_value=-20, max_value=20, step=1, key="flat_thrust_low")
            flat_thrust_mid = st.slider("Flatwing Thrust Bias Mid (%)", min_value=-20, max_value=20, step=1, key="flat_thrust_mid")
            flat_thrust_high = st.slider("Flatwing Thrust Bias High (%)", min_value=-20, max_value=20, step=1, key="flat_thrust_high")

            _flat_cdo_base, _flat_k_base = _baseline_cdo_k("Flatwing")
            _flat_cdo_line = st.empty()
            flat_drag_cdo_delta_pct = st.slider("Flatwing CDO Delta (%)", min_value=-10, max_value=10, step=1, key="flat_drag_cdo_delta_pct")
            try:
                _flat_cdo_line.caption(f"CDO: {_flat_cdo_base * (1.0 + float(flat_drag_cdo_delta_pct) / 100.0):.6f}  (base={_flat_cdo_base:.6f})")
            except Exception:
                pass
            _flat_k_line = st.empty()
            flat_drag_k_delta_pct = st.slider("Flatwing K Delta (%)", min_value=-10, max_value=10, step=1, key="flat_drag_k_delta_pct")
            try:
                _flat_k_line.caption(f"K: {_flat_k_base * (1.0 + float(flat_drag_k_delta_pct) / 100.0):.6f}  (base={_flat_k_base:.6f})")
            except Exception:
                pass

    else:
        st.info("Auto mode: enter per-mod real-world metrics. Biases will be derived heuristically.")
        st.caption("AutoBias calibration assumes MTOW departure.")
        col_auto_tam, col_auto_fw = st.columns(2)
        with col_auto_fw:
            st.markdown("**Flatwing Auto Inputs**")
            real_t_climb_min_f = st.number_input("Time to Climb (min)", min_value=0.0, value=30.0, step=0.5, key="real_t_climb_min_f")
            real_fuel_toc_lb_f = st.number_input("Fuel to TOC (lb)", min_value=0.0, value=600.0, step=10.0, key="real_fuel_toc_lb_f")
            real_cruise_pph_f = st.number_input("Cruise Fuel Burn (lb/hr)", min_value=0.0, value=700.0, step=10.0, key="real_cruise_pph_f")
            init_cruise_alt_ft_f = st.number_input("Initial Cruise Altitude (ft)", min_value=0, max_value=int(ceiling), value=min(int(ceiling), 35000), step=1000, key="init_cruise_alt_ft_f")
        with col_auto_tam:
            st.markdown("**Tamarack Auto Inputs**")
            real_t_climb_min_t = st.number_input("Time to Climb (min)", min_value=0.0, value=30.0, step=0.5, key="real_t_climb_min_t")
            real_fuel_toc_lb_t = st.number_input("Fuel to TOC (lb)", min_value=0.0, value=550.0, step=10.0, key="real_fuel_toc_lb_t")
            real_cruise_pph_t = st.number_input("Cruise Fuel Burn (lb/hr)", min_value=0.0, value=600.0, step=10.0, key="real_cruise_pph_t")
            init_cruise_alt_ft_t = st.number_input("Initial Cruise Altitude (ft)", min_value=0, max_value=int(ceiling), value=min(int(ceiling), 41000), step=1000, key="init_cruise_alt_ft_t")

        st.radio(
            "AutoBias targets",
            ["Engines", "Aerodynamics", "Both"],
            index=2,
            key="autobias_target",
            help="Engines: adjust SFC/Thrust only. Aerodynamics: adjust CDO/K only. Both: adjust all four."
        )

        try:
            delta_f = max(0, int(cruise_altitude_f) - int(init_cruise_alt_ft_f))
        except Exception:
            delta_f = 0
        try:
            delta_t = max(0, int(cruise_altitude_t) - int(init_cruise_alt_ft_t))
        except Exception:
            delta_t = 0
        scale_f = min(15, delta_f // 2000)
        scale_t = min(15, delta_t // 2000)
        flat_sfc_low = 0
        flat_sfc_mid = int(scale_f // 2)
        flat_sfc_high = int(scale_f)
        flat_thrust_low = 0
        flat_thrust_mid = -int(scale_f // 2)
        flat_thrust_high = -int(scale_f)
        tam_sfc_low = 0
        tam_sfc_mid = int(scale_t // 2)
        tam_sfc_high = int(scale_t)
        tam_thrust_low = 0
        tam_thrust_mid = -int(scale_t // 2)
        tam_thrust_high = -int(scale_t)

    if bias_mode == "Auto":
        applied = st.session_state.get('applied_biases', {})
        tam_defaults = {
            'sfc_low': st.session_state.get('tam_sfc_low', tam_sfc_low),
            'sfc_mid': st.session_state.get('tam_sfc_mid', tam_sfc_mid),
            'sfc_high': st.session_state.get('tam_sfc_high', tam_sfc_high),
            'thrust_low': st.session_state.get('tam_thrust_low', tam_thrust_low),
            'thrust_mid': st.session_state.get('tam_thrust_mid', tam_thrust_mid),
            'thrust_high': st.session_state.get('tam_thrust_high', tam_thrust_high),
            'drag_cdo_delta_pct': st.session_state.get('tam_drag_cdo_delta_pct', 0),
            'drag_k_delta_pct': st.session_state.get('tam_drag_k_delta_pct', 0),
        }
        flat_defaults = {
            'sfc_low': st.session_state.get('flat_sfc_low', flat_sfc_low),
            'sfc_mid': st.session_state.get('flat_sfc_mid', flat_sfc_mid),
            'sfc_high': st.session_state.get('flat_sfc_high', flat_sfc_high),
            'thrust_low': st.session_state.get('flat_thrust_low', flat_thrust_low),
            'thrust_mid': st.session_state.get('flat_thrust_mid', flat_thrust_mid),
            'thrust_high': st.session_state.get('flat_thrust_high', flat_thrust_high),
            'drag_cdo_delta_pct': st.session_state.get('flat_drag_cdo_delta_pct', 0),
            'drag_k_delta_pct': st.session_state.get('flat_drag_k_delta_pct', 0),
        }
        tam_applied = applied.get('Tamarack', tam_defaults)
        flat_applied = applied.get('Flatwing', flat_defaults)
        try:
            tam_sfc_low = int(tam_applied.get('sfc_low', tam_sfc_low))
            tam_sfc_mid = int(tam_applied.get('sfc_mid', tam_sfc_mid))
            tam_sfc_high = int(tam_applied.get('sfc_high', tam_sfc_high))
            tam_thrust_low = int(tam_applied.get('thrust_low', tam_thrust_low))
            tam_thrust_mid = int(tam_applied.get('thrust_mid', tam_thrust_mid))
            tam_thrust_high = int(tam_applied.get('thrust_high', tam_thrust_high))
        except Exception:
            pass
        try:
            flat_sfc_low = int(flat_applied.get('sfc_low', flat_sfc_low))
            flat_sfc_mid = int(flat_applied.get('sfc_mid', flat_sfc_mid))
            flat_sfc_high = int(flat_applied.get('sfc_high', flat_sfc_high))
            flat_thrust_low = int(flat_applied.get('thrust_low', flat_thrust_low))
            flat_thrust_mid = int(flat_applied.get('thrust_mid', flat_thrust_mid))
            flat_thrust_high = int(flat_applied.get('thrust_high', flat_thrust_high))
        except Exception:
            pass
        col_bias_left, col_bias_right = st.columns(2)

        def _bias_md(sfc_low, sfc_mid, sfc_high, thrust_low, thrust_mid, thrust_high) -> str:
            return (
                f"- **SFC Low (%):** {sfc_low}\n"
                f"- **SFC Mid (%):** {sfc_mid}\n"
                f"- **SFC High (%):** {sfc_high}\n"
                f"- **Thrust Low (%):** {thrust_low}\n"
                f"- **Thrust Mid (%):** {thrust_mid}\n"
                f"- **Thrust High (%):** {thrust_high}"
            )

        with col_bias_left:
            st.markdown("**Tamarack Biases (Applied)**")
            tam_bias_placeholder = st.empty()
            tam_bias_placeholder.markdown(_bias_md(
                tam_sfc_low, tam_sfc_mid, tam_sfc_high,
                tam_thrust_low, tam_thrust_mid, tam_thrust_high
            ))
            tam_report_placeholder = st.empty()
            try:
                _r = st.session_state.get('auto_bias_report_tamarack')
                if _r:
                    tam_report_placeholder.caption(_r)
            except Exception:
                pass

            _tam_cdo_base, _tam_k_base = _baseline_cdo_k("Tamarack")
            _tam_cdo_line = st.empty()
            tam_drag_cdo_delta_pct = st.slider(
                "Tamarack CDO Delta (%)",
                min_value=-10,
                max_value=10,
                step=1,
                value=int(tam_applied.get('drag_cdo_delta_pct', 0) or 0),
                key="tam_drag_cdo_delta_pct_applied",
                disabled=True,
            )
            try:
                _tam_cdo_line.caption(f"CDO: {_tam_cdo_base * (1.0 + float(tam_drag_cdo_delta_pct) / 100.0):.6f}  (base={_tam_cdo_base:.6f})")
            except Exception:
                pass
            _tam_k_line = st.empty()
            tam_drag_k_delta_pct = st.slider(
                "Tamarack K Delta (%)",
                min_value=-10,
                max_value=10,
                step=1,
                value=int(tam_applied.get('drag_k_delta_pct', 0) or 0),
                key="tam_drag_k_delta_pct_applied",
                disabled=True,
            )
            try:
                _tam_k_line.caption(f"K: {_tam_k_base * (1.0 + float(tam_drag_k_delta_pct) / 100.0):.6f}  (base={_tam_k_base:.6f})")
            except Exception:
                pass
        with col_bias_right:
            st.markdown("**Flatwing Biases (Applied)**")
            flat_bias_placeholder = st.empty()
            flat_bias_placeholder.markdown(_bias_md(
                flat_sfc_low, flat_sfc_mid, flat_sfc_high,
                flat_thrust_low, flat_thrust_mid, flat_thrust_high
            ))
            flat_report_placeholder = st.empty()
            try:
                _r = st.session_state.get('auto_bias_report_flatwing')
                if _r:
                    flat_report_placeholder.caption(_r)
            except Exception:
                pass

            _flat_cdo_base, _flat_k_base = _baseline_cdo_k("Flatwing")
            _flat_cdo_line = st.empty()
            flat_drag_cdo_delta_pct = st.slider(
                "Flatwing CDO Delta (%)",
                min_value=-10,
                max_value=10,
                step=1,
                value=int(flat_applied.get('drag_cdo_delta_pct', 0) or 0),
                key="flat_drag_cdo_delta_pct_applied",
                disabled=True,
            )
            try:
                _flat_cdo_line.caption(f"CDO: {_flat_cdo_base * (1.0 + float(flat_drag_cdo_delta_pct) / 100.0):.6f}  (base={_flat_cdo_base:.6f})")
            except Exception:
                pass
            _flat_k_line = st.empty()
            flat_drag_k_delta_pct = st.slider(
                "Flatwing K Delta (%)",
                min_value=-10,
                max_value=10,
                step=1,
                value=int(flat_applied.get('drag_k_delta_pct', 0) or 0),
                key="flat_drag_k_delta_pct_applied",
                disabled=True,
            )
            try:
                _flat_k_line.caption(f"K: {_flat_k_base * (1.0 + float(flat_drag_k_delta_pct) / 100.0):.6f}  (base={_flat_k_base:.6f})")
            except Exception:
                pass

    # Cruise mode selection
    cruise_mode = st.radio(
        "Cruise Mode",
        ["MCT (Max Thrust)", "Max Range", "Max Endurance"],
        index=0,
        format_func=lambda x: {"Max Range": "Max Range speed (LRC)", "Max Endurance": "Max Endurance Speed"}.get(x, x),
        help="Above 10,000 ft: set cruise speed by objective. For MCT, hold max thrust. For Max Range/Endurance, target optimal CL/CD with V >= 1.2*Vs."
    )

    run_all_modes = st.sidebar.checkbox("Run All Modes", value=False, help="Run MCT, LRC, and Max Endurance; show detailed results for the selected mode above.")
    v1_cut_enabled = st.sidebar.checkbox("Enable V1 Cut Simulation (Single Engine)", value=False)
    write_output_file = st.sidebar.checkbox("Write Output CSV File", value=True)
    fuel_cost_per_gal = st.sidebar.number_input("Fuel Cost ($/gal)", min_value=0.0, value=5.0, step=0.1)

    main.markdown("---")
    run_clicked = main.button("Run Simulation", type="primary")
    if not run_clicked:
        if 'last_run' not in st.session_state:
            main.info("Adjust parameters, then click 'Run Simulation' to execute.")
            st.stop()
        lr = st.session_state['last_run']
        try:
            with main:
                display_simulation_results(
                    lr.get('tamarack_data', pd.DataFrame()), lr.get('tamarack_results', {}),
                    lr.get('flatwing_data', pd.DataFrame()), lr.get('flatwing_results', {}),
                    v1_cut_enabled,
                    lr.get('dep_lat', 0.0), lr.get('dep_lon', 0.0), lr.get('arr_lat', 0.0), lr.get('arr_lon', 0.0),
                    lr.get('distance_nm', 0.0), lr.get('bearing_deg', 0.0),
                    winds_temps_source,
                    isa_dev,
                    (lr.get('cruise_altitude_f') if wing_type == "Flatwing" else lr.get('cruise_altitude_t')),
                    lr.get('dep_airport_code', ''),
                    lr.get('arr_airport_code', ''),
                    (lr.get('fuel_f') if wing_type == "Flatwing" else lr.get('fuel_t')),
                    report_output_dir=lr.get('report_output_dir'),
                    weight_df_flatwing=lr.get('weight_df_f'),
                    weight_df_tamarack=lr.get('weight_df_t'),
                    weight_df_single=lr.get('weight_df_single'),
                    modes_summary_df=lr.get('modes_summary_df'),
                    fuel_cost_per_gal=fuel_cost_per_gal
                )
        except Exception as e:
            main.exception(e)
        st.stop()

    try:
        del st.session_state['last_run']
    except Exception:
        pass

    _run_status = main.empty()
    try:
        _run_status.write("Running...")
    except Exception:
        pass

    if bias_mode == "Auto" and ("Tamarack" in mods_available or "Flatwing" in mods_available):
        _auto_status = main.empty()
        try:
            _auto_status.write(
                "Auto bias calibration running... "
                f"(aircraft={aircraft_model}, wing_type={wing_type}, cruise_mode={cruise_mode}, "
                f"tgt_climb={float(real_t_climb_min_t) if 'real_t_climb_min_t' in locals() else 0:.1f} min, "
                f"tgt_cruise={float(real_cruise_pph_t) if 'real_cruise_pph_t' in locals() else 0:.0f} lb/hr)"
            )
        except Exception:
            pass

        def _clamp_bias(v):
            try:
                return int(max(-20, min(20, round(float(v)))))
            except Exception:
                return 0

        def _pph_from_results(r: dict) -> float:
            try:
                fuel_lb = float(r.get("Cruise Fuel (lb)", 0) or 0)
                time_min = float(r.get("Cruise Time (min)", 0) or 0)
                if time_min <= 0:
                    fuel_lb = float(r.get("Total Fuel Burned (lb)", 0) or 0)
                    time_min = float(r.get("Total Time (min)", 0) or 0)
                    if time_min <= 0:
                        return 0.0
                return fuel_lb / (time_min / 60.0)
            except Exception:
                return 0.0

        def _pph_from_run(r: dict, df: pd.DataFrame | None) -> float:
            try:
                if isinstance(df, pd.DataFrame) and (not df.empty) and ("Fuel Flow (lb/hr)" in df.columns) and ("Segment" in df.columns):
                    seg = df["Segment"]
                    mask = seg.isin([6, 7])
                    vals = pd.to_numeric(df.loc[mask, "Fuel Flow (lb/hr)"], errors="coerce")
                    vals = vals[vals > 0]
                    if vals.notna().any():
                        return float(vals.median())
            except Exception:
                pass
            return _pph_from_results(r)

        def _any_nonzero(d: dict) -> bool:
            try:
                _keys = ('sfc_low', 'sfc_mid', 'sfc_high', 'thrust_low', 'thrust_mid', 'thrust_high')
                try:
                    if str(st.session_state.get('autobias_target', 'Both')) in ('Aerodynamics', 'Both'):
                        _keys = _keys + ('drag_cdo_delta_pct', 'drag_k_delta_pct')
                except Exception:
                    pass

                for k in _keys:
                    if int(d.get(k, 0) or 0) != 0:
                        return True
                return False
            except Exception:
                return False

        def _clamp_drag(v):
            try:
                return int(max(-10, min(10, round(float(v)))))
            except Exception:
                return 0

        def _mtow_payload_fuel(bow, mzfw, mtow, max_fuel, taxi_fuel, reserve_fuel, payload_guess, fuel_guess):
            try:
                bow = float(bow)
                mzfw = float(mzfw)
                mtow = float(mtow)
                max_fuel = float(max_fuel)
                taxi_fuel = float(taxi_fuel)
                reserve_fuel = float(reserve_fuel)
                payload_guess = float(payload_guess)
                fuel_guess = float(fuel_guess)
            except Exception:
                return payload_guess, fuel_guess

            max_payload = max(0.0, mzfw - bow)
            min_fuel = max(0.0, taxi_fuel + reserve_fuel)
            desired_payload_plus_fuel = mtow - bow + taxi_fuel

            payload = max(0.0, min(max_payload, payload_guess))
            fuel = desired_payload_plus_fuel - payload
            fuel = max(min_fuel, min(max_fuel, fuel))

            payload = desired_payload_plus_fuel - fuel
            payload = max(0.0, min(max_payload, payload))

            fuel = desired_payload_plus_fuel - payload
            fuel = max(min_fuel, min(max_fuel, fuel))

            return float(payload), float(fuel)

        _target = "Both"
        try:
            _target = str(st.session_state.get('autobias_target', 'Both') or 'Both')
        except Exception:
            _target = "Both"
        _adjust_engines = _target in ("Engines", "Both")
        _adjust_drag = _target in ("Aerodynamics", "Both")

        if "Tamarack" in mods_available:
            try:
                try:
                    _cal_payload_t, _cal_fuel_t = _mtow_payload_fuel(
                        tamarack_bow,
                        tamarack_mzfw,
                        tamarack_mtow,
                        tamarack_max_fuel,
                        taxi_fuel_t,
                        reserve_fuel_t,
                        payload_t,
                        fuel_t,
                    )
                except Exception:
                    _cal_payload_t, _cal_fuel_t = payload_t, fuel_t

                sfc_low = _clamp_bias(tam_sfc_low)
                sfc_mid = _clamp_bias(tam_sfc_mid)
                sfc_high = _clamp_bias(tam_sfc_high)
                thrust_low = _clamp_bias(tam_thrust_low)
                thrust_mid = _clamp_bias(tam_thrust_mid)
                thrust_high = _clamp_bias(tam_thrust_high)

                drag_cdo_delta = _clamp_drag(tam_drag_cdo_delta_pct)
                drag_k_delta = _clamp_drag(tam_drag_k_delta_pct)

                _last_tgt_climb = 0.0
                _last_sim_climb = 0.0
                _last_tgt_pph = 0.0
                _last_sim_pph = 0.0
                _last_tgt_toc = 0.0
                _last_sim_toc = 0.0

                for _ in range(3):
                    sim_df, sim_res, *_ = run_simulation(
                        dep_airport_code, arr_airport_code, aircraft_model, "Tamarack", takeoff_flap,
                        _cal_payload_t, _cal_fuel_t, taxi_fuel_t, reserve_fuel_t, cruise_altitude_t,
                        winds_temps_source, v1_cut_enabled, False, cruise_mode=cruise_mode,
                        sfc_bias_low=sfc_low, sfc_bias_mid=sfc_mid, sfc_bias_high=sfc_high,
                        thrust_bias_low=thrust_low, thrust_bias_mid=thrust_mid, thrust_bias_high=thrust_high,
                        bias_alt_mid=bias_alt_mid_ft,
                        drag_cdo_delta_pct=drag_cdo_delta,
                        drag_k_delta_pct=drag_k_delta
                    )

                    try:
                        tgt_climb = float(real_t_climb_min_t)
                    except Exception:
                        tgt_climb = 0.0
                    try:
                        sim_climb = float(sim_res.get("Climb Time (min)", 0) or 0)
                    except Exception:
                        sim_climb = 0.0

                    _last_tgt_climb = tgt_climb
                    _last_sim_climb = sim_climb

                    if tgt_climb > 0 and sim_climb > 0:
                        ratio = (sim_climb - tgt_climb) / tgt_climb
                        delta = _clamp_bias(ratio * 40)
                        if _adjust_engines and delta != 0:
                            thrust_low = _clamp_bias(thrust_low + delta)
                            thrust_mid = _clamp_bias(thrust_mid + delta)
                            thrust_high = _clamp_bias(thrust_high + delta)
                        if _adjust_drag:
                            d = _clamp_drag(-ratio * 20)
                            drag_cdo_delta = _clamp_drag(drag_cdo_delta + d)
                            drag_k_delta = _clamp_drag(drag_k_delta + _clamp_drag(d * 0.5))

                    try:
                        tgt_pph = float(real_cruise_pph_t)
                    except Exception:
                        tgt_pph = 0.0
                    sim_pph = _pph_from_run(sim_res, sim_df)

                    _last_tgt_pph = tgt_pph
                    _last_sim_pph = sim_pph

                    if tgt_pph > 0 and sim_pph > 0:
                        ratio = (sim_pph - tgt_pph) / tgt_pph
                        delta = _clamp_bias(ratio * 40)
                        delta = _clamp_bias(-delta)
                        if _adjust_engines and delta != 0:
                            sfc_high = _clamp_bias(sfc_high + delta)
                            sfc_mid = _clamp_bias(sfc_mid + int(round(delta * 0.5)))
                        if _adjust_drag:
                            d = _clamp_drag(-ratio * 20)
                            drag_cdo_delta = _clamp_drag(drag_cdo_delta + d)
                            drag_k_delta = _clamp_drag(drag_k_delta + _clamp_drag(d * 0.3))

                    try:
                        tgt_toc = float(real_fuel_toc_lb_t)
                    except Exception:
                        tgt_toc = 0.0
                    try:
                        sim_toc = float(sim_res.get("Climb Fuel (lb)", 0) or 0)
                    except Exception:
                        sim_toc = 0.0

                    _last_tgt_toc = tgt_toc
                    _last_sim_toc = sim_toc

                    if tgt_toc > 0 and sim_toc > 0:
                        ratio = (sim_toc - tgt_toc) / tgt_toc
                        delta = _clamp_bias(ratio * 40)
                        delta = _clamp_bias(-delta)
                        if _adjust_engines and delta != 0:
                            sfc_low = _clamp_bias(sfc_low + delta)
                            sfc_mid = _clamp_bias(sfc_mid + int(round(delta * 0.5)))
                        if _adjust_drag:
                            d = _clamp_drag(-ratio * 15)
                            drag_cdo_delta = _clamp_drag(drag_cdo_delta + d)
                            drag_k_delta = _clamp_drag(drag_k_delta + _clamp_drag(d * 0.2))

                tam_sfc_low, tam_sfc_mid, tam_sfc_high = sfc_low, sfc_mid, sfc_high
                tam_thrust_low, tam_thrust_mid, tam_thrust_high = thrust_low, thrust_mid, thrust_high
                tam_drag_cdo_delta_pct, tam_drag_k_delta_pct = drag_cdo_delta, drag_k_delta
                applied = st.session_state.get('applied_biases', {})
                prev_tam = applied.get('Tamarack')
                new_tam = {
                    'sfc_low': tam_sfc_low, 'sfc_mid': tam_sfc_mid, 'sfc_high': tam_sfc_high,
                    'thrust_low': tam_thrust_low, 'thrust_mid': tam_thrust_mid, 'thrust_high': tam_thrust_high,
                    'drag_cdo_delta_pct': int(tam_drag_cdo_delta_pct),
                    'drag_k_delta_pct': int(tam_drag_k_delta_pct),
                }

                keep_previous = False
                try:
                    if (not _any_nonzero(new_tam)) and isinstance(prev_tam, dict) and _any_nonzero(prev_tam):
                        if (_last_sim_climb == 0.0 and _last_sim_pph == 0.0) or (_last_tgt_climb == 0.0 and _last_tgt_pph == 0.0):
                            keep_previous = True
                except Exception:
                    keep_previous = False

                if keep_previous:
                    try:
                        tam_sfc_low = int(prev_tam.get('sfc_low', tam_sfc_low))
                        tam_sfc_mid = int(prev_tam.get('sfc_mid', tam_sfc_mid))
                        tam_sfc_high = int(prev_tam.get('sfc_high', tam_sfc_high))
                        tam_thrust_low = int(prev_tam.get('thrust_low', tam_thrust_low))
                        tam_thrust_mid = int(prev_tam.get('thrust_mid', tam_thrust_mid))
                        tam_thrust_high = int(prev_tam.get('thrust_high', tam_thrust_high))
                        tam_drag_cdo_delta_pct = int(prev_tam.get('drag_cdo_delta_pct', tam_drag_cdo_delta_pct))
                        tam_drag_k_delta_pct = int(prev_tam.get('drag_k_delta_pct', tam_drag_k_delta_pct))
                    except Exception:
                        pass
                else:
                    applied['Tamarack'] = new_tam
                    st.session_state['applied_biases'] = applied
                try:
                    st.session_state['tam_sfc_low'] = int(tam_sfc_low)
                    st.session_state['tam_sfc_mid'] = int(tam_sfc_mid)
                    st.session_state['tam_sfc_high'] = int(tam_sfc_high)
                    st.session_state['tam_thrust_low'] = int(tam_thrust_low)
                    st.session_state['tam_thrust_mid'] = int(tam_thrust_mid)
                    st.session_state['tam_thrust_high'] = int(tam_thrust_high)
                    st.session_state['tam_drag_cdo_delta_pct'] = int(tam_drag_cdo_delta_pct)
                    st.session_state['tam_drag_k_delta_pct'] = int(tam_drag_k_delta_pct)
                    st.session_state['tam_drag_cdo_delta_pct_applied'] = int(tam_drag_cdo_delta_pct)
                    st.session_state['tam_drag_k_delta_pct_applied'] = int(tam_drag_k_delta_pct)
                except Exception:
                    pass

                try:
                    if '_tam_cdo_line' in locals():
                        _tam_cdo_base, _tam_k_base = _baseline_cdo_k("Tamarack")
                        _tam_cdo_line.caption(f"CDO: {_tam_cdo_base * (1.0 + float(tam_drag_cdo_delta_pct) / 100.0):.6f}  (base={_tam_cdo_base:.6f})")
                    if '_tam_k_line' in locals():
                        _tam_cdo_base, _tam_k_base = _baseline_cdo_k("Tamarack")
                        _tam_k_line.caption(f"K: {_tam_k_base * (1.0 + float(tam_drag_k_delta_pct) / 100.0):.6f}  (base={_tam_k_base:.6f})")
                except Exception:
                    pass
                try:
                    st.session_state['auto_bias_report_tamarack'] = (
                        f"Last Tamarack auto-bias: tgt climb={_last_tgt_climb:.1f} min, sim climb={_last_sim_climb:.1f} min; "
                        f"tgt cruise={_last_tgt_pph:.0f} lb/hr, sim cruise={_last_sim_pph:.0f} lb/hr; "
                        f"tgt TOC fuel={_last_tgt_toc:.0f} lb, sim TOC fuel={_last_sim_toc:.0f} lb; "
                        f"SFC=({tam_sfc_low},{tam_sfc_mid},{tam_sfc_high})%, Thrust=({tam_thrust_low},{tam_thrust_mid},{tam_thrust_high})%, "
                        f"DragΔ=(CDO {tam_drag_cdo_delta_pct}%, K {tam_drag_k_delta_pct}%)"
                    )
                except Exception:
                    pass
                try:
                    _auto_status.write(
                        "Auto bias complete. "
                        f"Target climb={_last_tgt_climb:.1f} min, Sim climb={_last_sim_climb:.1f} min; "
                        f"Target cruise={_last_tgt_pph:.0f} lb/hr, Sim cruise={_last_sim_pph:.0f} lb/hr; "
                        f"Target TOC fuel={_last_tgt_toc:.0f} lb, Sim TOC fuel={_last_sim_toc:.0f} lb. "
                        f"Tamarack SFC mid={tam_sfc_mid}%, Thrust mid={tam_thrust_mid}%, "
                        f"DragΔ=(CDO {tam_drag_cdo_delta_pct}%, K {tam_drag_k_delta_pct}%). "
                        f"Targets={_target}"
                    )
                except Exception:
                    pass

                try:
                    if 'tam_bias_placeholder' in locals():
                        tam_bias_placeholder.markdown(_bias_md(
                            tam_sfc_low, tam_sfc_mid, tam_sfc_high,
                            tam_thrust_low, tam_thrust_mid, tam_thrust_high
                        ))
                except Exception:
                    pass

                try:
                    if 'tam_report_placeholder' in locals():
                        _r = st.session_state.get('auto_bias_report_tamarack')
                        if _r:
                            tam_report_placeholder.caption(_r)
                except Exception:
                    pass
            except Exception as e:
                main.exception(e)

        if "Flatwing" in mods_available:
            try:
                try:
                    _cal_payload_f, _cal_fuel_f = _mtow_payload_fuel(
                        flatwing_bow,
                        flatwing_mzfw,
                        mtow,
                        max_fuel,
                        taxi_fuel_f,
                        reserve_fuel_f,
                        payload_f,
                        fuel_f,
                    )
                except Exception:
                    _cal_payload_f, _cal_fuel_f = payload_f, fuel_f

                sfc_low = _clamp_bias(flat_sfc_low)
                sfc_mid = _clamp_bias(flat_sfc_mid)
                sfc_high = _clamp_bias(flat_sfc_high)
                thrust_low = _clamp_bias(flat_thrust_low)
                thrust_mid = _clamp_bias(flat_thrust_mid)
                thrust_high = _clamp_bias(flat_thrust_high)

                drag_cdo_delta = _clamp_drag(flat_drag_cdo_delta_pct)
                drag_k_delta = _clamp_drag(flat_drag_k_delta_pct)

                _last_tgt_climb_f = 0.0
                _last_sim_climb_f = 0.0
                _last_tgt_pph_f = 0.0
                _last_sim_pph_f = 0.0
                _last_tgt_toc_f = 0.0
                _last_sim_toc_f = 0.0

                for _ in range(3):
                    sim_df, sim_res, *_ = run_simulation(
                        dep_airport_code, arr_airport_code, aircraft_model, "Flatwing", takeoff_flap,
                        _cal_payload_f, _cal_fuel_f, taxi_fuel_f, reserve_fuel_f, cruise_altitude_f,
                        winds_temps_source, v1_cut_enabled, False, cruise_mode=cruise_mode,
                        sfc_bias_low=sfc_low, sfc_bias_mid=sfc_mid, sfc_bias_high=sfc_high,
                        thrust_bias_low=thrust_low, thrust_bias_mid=thrust_mid, thrust_bias_high=thrust_high,
                        bias_alt_mid=bias_alt_mid_ft,
                        drag_cdo_delta_pct=drag_cdo_delta,
                        drag_k_delta_pct=drag_k_delta
                    )

                    try:
                        tgt_climb = float(real_t_climb_min_f)
                    except Exception:
                        tgt_climb = 0.0
                    try:
                        sim_climb = float(sim_res.get("Climb Time (min)", 0) or 0)
                    except Exception:
                        sim_climb = 0.0

                    _last_tgt_climb_f = tgt_climb
                    _last_sim_climb_f = sim_climb

                    if tgt_climb > 0 and sim_climb > 0:
                        ratio = (sim_climb - tgt_climb) / tgt_climb
                        delta = _clamp_bias(ratio * 40)
                        if _adjust_engines and delta != 0:
                            thrust_low = _clamp_bias(thrust_low + delta)
                            thrust_mid = _clamp_bias(thrust_mid + delta)
                            thrust_high = _clamp_bias(thrust_high + delta)
                        if _adjust_drag:
                            d = _clamp_drag(-ratio * 20)
                            drag_cdo_delta = _clamp_drag(drag_cdo_delta + d)
                            drag_k_delta = _clamp_drag(drag_k_delta + _clamp_drag(d * 0.5))

                    try:
                        tgt_pph = float(real_cruise_pph_f)
                    except Exception:
                        tgt_pph = 0.0
                    sim_pph = _pph_from_run(sim_res, sim_df)

                    _last_tgt_pph_f = tgt_pph
                    _last_sim_pph_f = sim_pph

                    if tgt_pph > 0 and sim_pph > 0:
                        ratio = (sim_pph - tgt_pph) / tgt_pph
                        delta = _clamp_bias(ratio * 40)
                        delta = _clamp_bias(-delta)
                        if _adjust_engines and delta != 0:
                            sfc_high = _clamp_bias(sfc_high + delta)
                            sfc_mid = _clamp_bias(sfc_mid + int(round(delta * 0.5)))
                        if _adjust_drag:
                            d = _clamp_drag(-ratio * 20)
                            drag_cdo_delta = _clamp_drag(drag_cdo_delta + d)
                            drag_k_delta = _clamp_drag(drag_k_delta + _clamp_drag(d * 0.3))

                    try:
                        tgt_toc = float(real_fuel_toc_lb_f)
                    except Exception:
                        tgt_toc = 0.0
                    try:
                        sim_toc = float(sim_res.get("Climb Fuel (lb)", 0) or 0)
                    except Exception:
                        sim_toc = 0.0

                    _last_tgt_toc_f = tgt_toc
                    _last_sim_toc_f = sim_toc

                    if tgt_toc > 0 and sim_toc > 0:
                        ratio = (sim_toc - tgt_toc) / tgt_toc
                        delta = _clamp_bias(ratio * 40)
                        delta = _clamp_bias(-delta)
                        if _adjust_engines and delta != 0:
                            sfc_low = _clamp_bias(sfc_low + delta)
                            sfc_mid = _clamp_bias(sfc_mid + int(round(delta * 0.5)))
                        if _adjust_drag:
                            d = _clamp_drag(-ratio * 15)
                            drag_cdo_delta = _clamp_drag(drag_cdo_delta + d)
                            drag_k_delta = _clamp_drag(drag_k_delta + _clamp_drag(d * 0.2))

                flat_sfc_low, flat_sfc_mid, flat_sfc_high = sfc_low, sfc_mid, sfc_high
                flat_thrust_low, flat_thrust_mid, flat_thrust_high = thrust_low, thrust_mid, thrust_high
                flat_drag_cdo_delta_pct, flat_drag_k_delta_pct = drag_cdo_delta, drag_k_delta
                applied = st.session_state.get('applied_biases', {})
                prev_flat = applied.get('Flatwing')
                new_flat = {
                    'sfc_low': flat_sfc_low, 'sfc_mid': flat_sfc_mid, 'sfc_high': flat_sfc_high,
                    'thrust_low': flat_thrust_low, 'thrust_mid': flat_thrust_mid, 'thrust_high': flat_thrust_high,
                    'drag_cdo_delta_pct': int(flat_drag_cdo_delta_pct),
                    'drag_k_delta_pct': int(flat_drag_k_delta_pct),
                }

                keep_previous = False
                try:
                    if (not _any_nonzero(new_flat)) and isinstance(prev_flat, dict) and _any_nonzero(prev_flat):
                        if (_last_sim_climb_f == 0.0 and _last_sim_pph_f == 0.0) or (_last_tgt_climb_f == 0.0 and _last_tgt_pph_f == 0.0):
                            keep_previous = True
                except Exception:
                    keep_previous = False

                if keep_previous:
                    try:
                        flat_sfc_low = int(prev_flat.get('sfc_low', flat_sfc_low))
                        flat_sfc_mid = int(prev_flat.get('sfc_mid', flat_sfc_mid))
                        flat_sfc_high = int(prev_flat.get('sfc_high', flat_sfc_high))
                        flat_thrust_low = int(prev_flat.get('thrust_low', flat_thrust_low))
                        flat_thrust_mid = int(prev_flat.get('thrust_mid', flat_thrust_mid))
                        flat_thrust_high = int(prev_flat.get('thrust_high', flat_thrust_high))
                        flat_drag_cdo_delta_pct = int(prev_flat.get('drag_cdo_delta_pct', flat_drag_cdo_delta_pct))
                        flat_drag_k_delta_pct = int(prev_flat.get('drag_k_delta_pct', flat_drag_k_delta_pct))
                    except Exception:
                        pass
                else:
                    applied['Flatwing'] = new_flat
                    st.session_state['applied_biases'] = applied

                try:
                    st.session_state['flat_sfc_low'] = int(flat_sfc_low)
                    st.session_state['flat_sfc_mid'] = int(flat_sfc_mid)
                    st.session_state['flat_sfc_high'] = int(flat_sfc_high)
                    st.session_state['flat_thrust_low'] = int(flat_thrust_low)
                    st.session_state['flat_thrust_mid'] = int(flat_thrust_mid)
                    st.session_state['flat_thrust_high'] = int(flat_thrust_high)
                    st.session_state['flat_drag_cdo_delta_pct'] = int(flat_drag_cdo_delta_pct)
                    st.session_state['flat_drag_k_delta_pct'] = int(flat_drag_k_delta_pct)
                    st.session_state['flat_drag_cdo_delta_pct_applied'] = int(flat_drag_cdo_delta_pct)
                    st.session_state['flat_drag_k_delta_pct_applied'] = int(flat_drag_k_delta_pct)
                except Exception:
                    pass

                try:
                    st.session_state['auto_bias_report_flatwing'] = (
                        f"Last Flatwing auto-bias: tgt climb={_last_tgt_climb_f:.1f} min, sim climb={_last_sim_climb_f:.1f} min; "
                        f"tgt cruise={_last_tgt_pph_f:.0f} lb/hr, sim cruise={_last_sim_pph_f:.0f} lb/hr; "
                        f"tgt TOC fuel={_last_tgt_toc_f:.0f} lb, sim TOC fuel={_last_sim_toc_f:.0f} lb; "
                        f"SFC=({flat_sfc_low},{flat_sfc_mid},{flat_sfc_high})%, Thrust=({flat_thrust_low},{flat_thrust_mid},{flat_thrust_high})%, "
                        f"DragΔ=(CDO {flat_drag_cdo_delta_pct}%, K {flat_drag_k_delta_pct}%)"
                    )
                except Exception:
                    pass

                try:
                    if 'flat_bias_placeholder' in locals():
                        flat_bias_placeholder.markdown(_bias_md(
                            flat_sfc_low, flat_sfc_mid, flat_sfc_high,
                            flat_thrust_low, flat_thrust_mid, flat_thrust_high
                        ))
                except Exception:
                    pass

                try:
                    if 'flat_report_placeholder' in locals():
                        _r = st.session_state.get('auto_bias_report_flatwing')
                        if _r:
                            flat_report_placeholder.caption(_r)
                except Exception:
                    pass
            except Exception as e:
                main.exception(e)

    def _clamp_drag_outer(v):
        try:
            return int(max(-10, min(10, int(v))))
        except Exception:
            return 0

    tam_drag_cdo_to_use = _clamp_drag_outer(st.session_state.get('tam_drag_cdo_delta_pct', 0) or 0)
    tam_drag_k_to_use = _clamp_drag_outer(st.session_state.get('tam_drag_k_delta_pct', 0) or 0)
    flat_drag_cdo_to_use = _clamp_drag_outer(st.session_state.get('flat_drag_cdo_delta_pct', 0) or 0)
    flat_drag_k_to_use = _clamp_drag_outer(st.session_state.get('flat_drag_k_delta_pct', 0) or 0)
    if bias_mode == "Auto":
        try:
            _applied = st.session_state.get('applied_biases', {})
            tam_drag_cdo_to_use = _clamp_drag_outer(_applied.get('Tamarack', {}).get('drag_cdo_delta_pct', tam_drag_cdo_to_use) or tam_drag_cdo_to_use)
            tam_drag_k_to_use = _clamp_drag_outer(_applied.get('Tamarack', {}).get('drag_k_delta_pct', tam_drag_k_to_use) or tam_drag_k_to_use)
            flat_drag_cdo_to_use = _clamp_drag_outer(_applied.get('Flatwing', {}).get('drag_cdo_delta_pct', flat_drag_cdo_to_use) or flat_drag_cdo_to_use)
            flat_drag_k_to_use = _clamp_drag_outer(_applied.get('Flatwing', {}).get('drag_k_delta_pct', flat_drag_k_to_use) or flat_drag_k_to_use)
        except Exception:
            pass

    chosen_mode = cruise_mode
    modes_summary_df = None
    tamarack_output_file = ""
    flatwing_output_file = ""

    if run_all_modes:
        modes_to_run = ["MCT (Max Thrust)", "Max Range", "Max Endurance"]
        ordered_modes = [m for m in modes_to_run if m == chosen_mode] + [m for m in modes_to_run if m != chosen_mode]
        results_by_mode = {}

        try:
            _mods_count = 1
            if wing_type == "Comparison":
                _mods_count = int((1 if "Tamarack" in mods_available else 0) + (1 if "Flatwing" in mods_available else 0))
                if _mods_count <= 0:
                    _mods_count = 1
            _mode_total_steps = max(1, len(modes_to_run) * _mods_count)
            _mode_step = 0
            _mode_progress = main.progress(0)
            _mode_status = main.empty()
        except Exception:
            _mode_total_steps = 1
            _mode_step = 0
            _mode_progress = None
            _mode_status = None

        chosen_coords = None
        first_coords = None

        for mode in modes_to_run:
            res = {}
            if wing_type == "Comparison":
                if "Tamarack" in mods_available:
                    try:
                        if _mode_status is not None:
                            _mode_status.write(f"Running {mode} - Tamarack ({_mode_step + 1}/{_mode_total_steps})")
                    except Exception:
                        pass
                    t_data, t_results, dep_lat, dep_lon, arr_lat, arr_lon, t_out = run_simulation(
                        dep_airport_code, arr_airport_code, aircraft_model, "Tamarack", takeoff_flap,
                        payload_t, fuel_t, taxi_fuel_t, reserve_fuel_t, cruise_altitude_t,
                        winds_temps_source, v1_cut_enabled, write_output_file, cruise_mode=mode,
                        sfc_bias_low=tam_sfc_low, sfc_bias_mid=tam_sfc_mid, sfc_bias_high=tam_sfc_high,
                        thrust_bias_low=tam_thrust_low, thrust_bias_mid=tam_thrust_mid, thrust_bias_high=tam_thrust_high,
                        bias_alt_mid=bias_alt_mid_ft,
                        drag_cdo_delta_pct=tam_drag_cdo_to_use,
                        drag_k_delta_pct=tam_drag_k_to_use)
                    res["Tamarack"] = (t_data, t_results, t_out)
                    if first_coords is None:
                        first_coords = (dep_lat, dep_lon, arr_lat, arr_lon)
                    if mode == chosen_mode:
                        chosen_coords = (dep_lat, dep_lon, arr_lat, arr_lon)
                    _mode_step += 1
                    try:
                        if _mode_progress is not None:
                            _mode_progress.progress(min(1.0, _mode_step / max(1, _mode_total_steps)))
                    except Exception:
                        pass
                if "Flatwing" in mods_available:
                    try:
                        if _mode_status is not None:
                            _mode_status.write(f"Running {mode} - Flatwing ({_mode_step + 1}/{_mode_total_steps})")
                    except Exception:
                        pass
                    f_data, f_results, dep_lat, dep_lon, arr_lat, arr_lon, f_out = run_simulation(
                        dep_airport_code, arr_airport_code, aircraft_model, "Flatwing", takeoff_flap,
                        payload_f, fuel_f, taxi_fuel_f, reserve_fuel_f, cruise_altitude_f,
                        winds_temps_source, v1_cut_enabled, write_output_file, cruise_mode=mode,
                        sfc_bias_low=flat_sfc_low, sfc_bias_mid=flat_sfc_mid, sfc_bias_high=flat_sfc_high,
                        thrust_bias_low=flat_thrust_low, thrust_bias_mid=flat_thrust_mid, thrust_bias_high=flat_thrust_high,
                        bias_alt_mid=bias_alt_mid_ft,
                        drag_cdo_delta_pct=flat_drag_cdo_to_use,
                        drag_k_delta_pct=flat_drag_k_to_use)
                    res["Flatwing"] = (f_data, f_results, f_out)
                    if first_coords is None:
                        first_coords = (dep_lat, dep_lon, arr_lat, arr_lon)
                    if mode == chosen_mode and chosen_coords is None:
                        chosen_coords = (dep_lat, dep_lon, arr_lat, arr_lon)
                    _mode_step += 1
                    try:
                        if _mode_progress is not None:
                            _mode_progress.progress(min(1.0, _mode_step / max(1, _mode_total_steps)))
                    except Exception:
                        pass
            elif wing_type == "Tamarack":
                try:
                    if _mode_status is not None:
                        _mode_status.write(f"Running {mode} ({_mode_step + 1}/{_mode_total_steps})")
                except Exception:
                    pass
                t_data, t_results, dep_lat, dep_lon, arr_lat, arr_lon, t_out = run_simulation(
                    dep_airport_code, arr_airport_code, aircraft_model, "Tamarack", takeoff_flap,
                    payload_t, fuel_t, taxi_fuel_t, reserve_fuel_t, cruise_altitude_t,
                    winds_temps_source, v1_cut_enabled, write_output_file, cruise_mode=mode,
                    sfc_bias_low=tam_sfc_low, sfc_bias_mid=tam_sfc_mid, sfc_bias_high=tam_sfc_high,
                    thrust_bias_low=tam_thrust_low, thrust_bias_mid=tam_thrust_mid, thrust_bias_high=tam_thrust_high,
                    bias_alt_mid=bias_alt_mid_ft,
                    drag_cdo_delta_pct=tam_drag_cdo_to_use,
                    drag_k_delta_pct=tam_drag_k_to_use)
                res["Tamarack"] = (t_data, t_results, t_out)
                if first_coords is None:
                    first_coords = (dep_lat, dep_lon, arr_lat, arr_lon)
                if mode == chosen_mode:
                    chosen_coords = (dep_lat, dep_lon, arr_lat, arr_lon)
                _mode_step += 1
                try:
                    if _mode_progress is not None:
                        _mode_progress.progress(min(1.0, _mode_step / max(1, _mode_total_steps)))
                except Exception:
                    pass
            elif wing_type == "Flatwing":
                try:
                    if _mode_status is not None:
                        _mode_status.write(f"Running {mode} ({_mode_step + 1}/{_mode_total_steps})")
                except Exception:
                    pass
                f_data, f_results, dep_lat, dep_lon, arr_lat, arr_lon, f_out = run_simulation(
                    dep_airport_code, arr_airport_code, aircraft_model, "Flatwing", takeoff_flap,
                    payload_f, fuel_f, taxi_fuel_f, reserve_fuel_f, cruise_altitude_f,
                    winds_temps_source, v1_cut_enabled, write_output_file, cruise_mode=mode,
                    sfc_bias_low=flat_sfc_low, sfc_bias_mid=flat_sfc_mid, sfc_bias_high=flat_sfc_high,
                    thrust_bias_low=flat_thrust_low, thrust_bias_mid=flat_thrust_mid, thrust_bias_high=flat_thrust_high,
                    bias_alt_mid=bias_alt_mid_ft,
                    drag_cdo_delta_pct=flat_drag_cdo_to_use,
                    drag_k_delta_pct=flat_drag_k_to_use)
                res["Flatwing"] = (f_data, f_results, f_out)
                if first_coords is None:
                    first_coords = (dep_lat, dep_lon, arr_lat, arr_lon)
                if mode == chosen_mode:
                    chosen_coords = (dep_lat, dep_lon, arr_lat, arr_lon)
                _mode_step += 1
                try:
                    if _mode_progress is not None:
                        _mode_progress.progress(min(1.0, _mode_step / max(1, _mode_total_steps)))
                except Exception:
                    pass

            results_by_mode[mode] = res

        try:
            if _mode_status is not None:
                _mode_status.write("All modes complete.")
            if _mode_progress is not None:
                _mode_progress.progress(1.0)
        except Exception:
            pass

        if chosen_coords is None:
            chosen_coords = first_coords
        if chosen_coords is not None:
            dep_lat, dep_lon, arr_lat, arr_lon = chosen_coords

        rows = []
        def mode_short(m: str) -> str:
            return {"MCT (Max Thrust)": "MCT", "Max Range": "LRC", "Max Endurance": "Max End"}.get(m, m)

        if wing_type == "Comparison":
            for m in ordered_modes:
                t_res = results_by_mode.get(m, {}).get("Tamarack", (pd.DataFrame(), {}, ""))[1]
                f_res = results_by_mode.get(m, {}).get("Flatwing", (pd.DataFrame(), {}, ""))[1]
                t_burn = float(t_res.get("Total Fuel Burned (lb)", 0) or 0)
                f_burn = float(f_res.get("Total Fuel Burned (lb)", 0) or 0)
                t_time = float(t_res.get("Total Time (min)", 0) or 0)
                f_time = float(f_res.get("Total Time (min)", 0) or 0)
                lb_per_gal = 6.7
                savings_lb = f_burn - t_burn
                savings_gal = savings_lb / lb_per_gal if lb_per_gal > 0 else 0.0
                rows.append({
                    "Mode": mode_short(m),
                    "Chosen": (m == chosen_mode),
                    "Flatwing Fuel Used (lb)": f"{f_burn:,.0f}",
                    "Tamarack Fuel Used (lb)": f"{t_burn:,.0f}",
                    "Fuel Saved (lb)": f"{savings_lb:,.0f}",
                    "Fuel Saved (gal)": f"{savings_gal:,.1f}",
                    "Flatwing Time (min)": f"{f_time:,.0f}",
                    "Tamarack Time (min)": f"{t_time:,.0f}"
                })
        else:
            key = "Flatwing" if wing_type == "Flatwing" else "Tamarack"
            for m in ordered_modes:
                r = results_by_mode.get(m, {}).get(key, (pd.DataFrame(), {}, ""))[1]
                burn = float(r.get("Total Fuel Burned (lb)", 0) or 0)
                time_min = float(r.get("Total Time (min)", 0) or 0)
                rows.append({
                    "Mode": mode_short(m),
                    "Chosen": (m == chosen_mode),
                    "Fuel Used (lb)": f"{burn:,.0f}",
                    "Total Time (min)": f"{time_min:,.0f}"
                })
        try:
            modes_summary_df = pd.DataFrame(rows)
        except Exception:
            modes_summary_df = None

        chosen = results_by_mode.get(chosen_mode, {})
        if "Tamarack" in chosen:
            tamarack_data, tamarack_results, tamarack_output_file = chosen["Tamarack"]
        else:
            tamarack_data, tamarack_results, tamarack_output_file = pd.DataFrame(), {}, ""
        if "Flatwing" in chosen:
            flatwing_data, flatwing_results, flatwing_output_file = chosen["Flatwing"]
        else:
            flatwing_data, flatwing_results, flatwing_output_file = pd.DataFrame(), {}, ""
    else:
        tamarack_data, tamarack_results = pd.DataFrame(), {}
        flatwing_data, flatwing_results = pd.DataFrame(), {}
        if wing_type == "Comparison":
            if "Tamarack" in mods_available:
                tamarack_data, tamarack_results, dep_lat, dep_lon, arr_lat, arr_lon, tamarack_output_file = run_simulation(
                    dep_airport_code, arr_airport_code, aircraft_model, "Tamarack", takeoff_flap,
                    payload_t, fuel_t, taxi_fuel_t, reserve_fuel_t, cruise_altitude_t,
                    winds_temps_source, v1_cut_enabled, write_output_file, cruise_mode=cruise_mode,
                    sfc_bias_low=tam_sfc_low, sfc_bias_mid=tam_sfc_mid, sfc_bias_high=tam_sfc_high,
                    thrust_bias_low=tam_thrust_low, thrust_bias_mid=tam_thrust_mid, thrust_bias_high=tam_thrust_high,
                    bias_alt_mid=bias_alt_mid_ft,
                    drag_cdo_delta_pct=tam_drag_cdo_to_use,
                    drag_k_delta_pct=tam_drag_k_to_use)
            if "Flatwing" in mods_available:
                flatwing_data, flatwing_results, dep_lat, dep_lon, arr_lat, arr_lon, flatwing_output_file = run_simulation(
                    dep_airport_code, arr_airport_code, aircraft_model, "Flatwing", takeoff_flap,
                    payload_f, fuel_f, taxi_fuel_f, reserve_fuel_f, cruise_altitude_f,
                    winds_temps_source, v1_cut_enabled, write_output_file, cruise_mode=cruise_mode,
                    sfc_bias_low=flat_sfc_low, sfc_bias_mid=flat_sfc_mid, sfc_bias_high=flat_sfc_high,
                    thrust_bias_low=flat_thrust_low, thrust_bias_mid=flat_thrust_mid, thrust_bias_high=flat_thrust_high,
                    bias_alt_mid=bias_alt_mid_ft,
                    drag_cdo_delta_pct=flat_drag_cdo_to_use,
                    drag_k_delta_pct=flat_drag_k_to_use)
        elif wing_type == "Tamarack":
            tamarack_data, tamarack_results, dep_lat, dep_lon, arr_lat, arr_lon, tamarack_output_file = run_simulation(
                dep_airport_code, arr_airport_code, aircraft_model, "Tamarack", takeoff_flap,
                payload_t, fuel_t, taxi_fuel_t, reserve_fuel_t, cruise_altitude_t,
                winds_temps_source, v1_cut_enabled, write_output_file, cruise_mode=cruise_mode,
                sfc_bias_low=tam_sfc_low, sfc_bias_mid=tam_sfc_mid, sfc_bias_high=tam_sfc_high,
                thrust_bias_low=tam_thrust_low, thrust_bias_mid=tam_thrust_mid, thrust_bias_high=tam_thrust_high,
                bias_alt_mid=bias_alt_mid_ft,
                drag_cdo_delta_pct=tam_drag_cdo_to_use,
                drag_k_delta_pct=tam_drag_k_to_use)
        elif wing_type == "Flatwing":
            flatwing_data, flatwing_results, dep_lat, dep_lon, arr_lat, arr_lon, flatwing_output_file = run_simulation(
                dep_airport_code, arr_airport_code, aircraft_model, "Flatwing", takeoff_flap,
                payload_f, fuel_f, taxi_fuel_f, reserve_fuel_f, cruise_altitude_f,
                winds_temps_source, v1_cut_enabled, write_output_file, cruise_mode=cruise_mode,
                sfc_bias_low=flat_sfc_low, sfc_bias_mid=flat_sfc_mid, sfc_bias_high=flat_sfc_high,
                thrust_bias_low=flat_thrust_low, thrust_bias_mid=flat_thrust_mid, thrust_bias_high=flat_thrust_high,
                bias_alt_mid=bias_alt_mid_ft,
                drag_cdo_delta_pct=flat_drag_cdo_to_use,
                drag_k_delta_pct=flat_drag_k_to_use)# Display results
    try:
        distance_nm, bearing_deg = haversine_with_bearing(dep_lat, dep_lon, arr_lat, arr_lon)
    except Exception:
        distance_nm, bearing_deg = 0.0, 0.0
    main.markdown("---")
    main.header('Simulation Results')
    # Determine output folder for report (same as CSV)
    report_output_dir = None
    if 'tamarack_output_file' in locals():
        report_output_dir = os.path.dirname(tamarack_output_file)
    elif 'flatwing_output_file' in locals():
        report_output_dir = os.path.dirname(flatwing_output_file)

    # Build Weight Status tables for UI/PDF
    # Save last run state for persistence
    try:
        st.session_state['last_run'] = {
            'wing_type': wing_type,
            'dep_airport_code': dep_airport_code,
            'arr_airport_code': arr_airport_code,
            'dep_lat': dep_lat, 'dep_lon': dep_lon, 'arr_lat': arr_lat, 'arr_lon': arr_lon,
            'distance_nm': locals().get('distance_nm', 0.0), 'bearing_deg': locals().get('bearing_deg', 0.0),
            'tamarack_data': (tamarack_data if 'tamarack_data' in locals() else pd.DataFrame()),
            'tamarack_results': (tamarack_results if 'tamarack_results' in locals() else {}),
            'tamarack_output_file': (tamarack_output_file if 'tamarack_output_file' in locals() else ''),
            'flatwing_data': (flatwing_data if 'flatwing_data' in locals() else pd.DataFrame()),
            'flatwing_results': (flatwing_results if 'flatwing_results' in locals() else {}),
            'flatwing_output_file': (flatwing_output_file if 'flatwing_output_file' in locals() else ''),
            'report_output_dir': locals().get('report_output_dir', None),
            'weight_df_f': (weight_df_f if 'weight_df_f' in locals() else None),
            'weight_df_t': (weight_df_t if 'weight_df_t' in locals() else None),
            'weight_df_single': (weight_df if 'weight_df' in locals() else None),
            'modes_summary_df': (modes_summary_df if 'modes_summary_df' in locals() else None),
            'cruise_altitude_f': locals().get('cruise_altitude_f', None),
            'cruise_altitude_t': locals().get('cruise_altitude_t', None),
            'fuel_f': locals().get('fuel_f', None),
            'fuel_t': locals().get('fuel_t', None),
        }
    except Exception:
        pass
    try:
        def _fmt(x):
            try:
                return f"{float(x):,.0f}"
            except Exception:
                return str(x)

        # Flatwing
        if wing_type in ("Comparison", "Flatwing"):
            try:
                mission_fuel_f = max(0.0, float(fuel_f) - float(reserve_fuel_f) - float(taxi_fuel_f))
            except Exception:
                mission_fuel_f = 0.0
            try:
                zfw_calc_f = float(flatwing_bow) + float(payload_f)
                rw_calc_f = zfw_calc_f + float(fuel_f)
                tow_calc_f = rw_calc_f - float(taxi_fuel_f)
            except Exception:
                zfw_calc_f = rw_calc_f = tow_calc_f = 0.0
            max_payload_f = max(0.0, float(flatwing_mzfw) - float(flatwing_bow))
            weight_df_f = pd.DataFrame({
                'Component': ['BOW', 'Payload', 'Initial Fuel', 'Reserve Fuel', 'Taxi Fuel', 'Mission Fuel', 'ZFW', 'Ramp Weight', 'Takeoff Weight'],
                'Weight (lb)': [_fmt(flatwing_bow), _fmt(payload_f), _fmt(fuel_f), _fmt(reserve_fuel_f), _fmt(taxi_fuel_f), _fmt(mission_fuel_f), _fmt(zfw_calc_f), _fmt(rw_calc_f), _fmt(tow_calc_f)],
                'Max Weight (lb)': [_fmt(flatwing_bow), _fmt(max_payload_f), _fmt(max_fuel), _fmt(reserve_fuel_default), _fmt(taxi_fuel_default), _fmt(mission_fuel_f), _fmt(flatwing_mzfw), _fmt(mrw), _fmt(mtow)]
            })

        # Tamarack
        if wing_type in ("Comparison", "Tamarack") and tamarack_config is not None:
            try:
                mission_fuel_t = max(0.0, float(fuel_t) - float(reserve_fuel_t) - float(taxi_fuel_t))
            except Exception:
                mission_fuel_t = 0.0
            try:
                zfw_calc_t = float(tamarack_bow) + float(payload_t)
                rw_calc_t = zfw_calc_t + float(fuel_t)
                tow_calc_t = rw_calc_t - float(taxi_fuel_t)
            except Exception:
                zfw_calc_t = rw_calc_t = tow_calc_t = 0.0
            max_payload_t = max(0.0, float(tamarack_mzfw) - float(tamarack_bow))
            weight_df_t = pd.DataFrame({
                'Component': ['BOW', 'Payload', 'Initial Fuel', 'Reserve Fuel', 'Taxi Fuel', 'Mission Fuel', 'ZFW', 'Ramp Weight', 'Takeoff Weight'],
                'Weight (lb)': [_fmt(tamarack_bow), _fmt(payload_t), _fmt(fuel_t), _fmt(reserve_fuel_t), _fmt(taxi_fuel_t), _fmt(mission_fuel_t), _fmt(zfw_calc_t), _fmt(rw_calc_t), _fmt(tow_calc_t)],
                'Max Weight (lb)': [_fmt(tamarack_bow), _fmt(max_payload_t), _fmt(tamarack_max_fuel), _fmt(tamarack_reserve_fuel_default), _fmt(tamarack_taxi_fuel_default), _fmt(mission_fuel_t), _fmt(tamarack_mzfw), _fmt(tamarack_mrw), _fmt(tamarack_mtow)]
            })

        # Single mode alias
        if wing_type == "Flatwing" and 'weight_df_f' in locals():
            weight_df = weight_df_f
        elif wing_type == "Tamarack" and 'weight_df_t' in locals():
            weight_df = weight_df_t
    except Exception:
        pass

    with main:
        display_simulation_results(
            tamarack_data, tamarack_results,
            flatwing_data, flatwing_results,
            v1_cut_enabled,
            dep_lat, dep_lon, arr_lat, arr_lon,
            distance_nm, bearing_deg,
            winds_temps_source,
            isa_dev,
            (cruise_altitude_f if wing_type == "Flatwing" else cruise_altitude_t),
            dep_airport_code,
            arr_airport_code,
            (fuel_f if wing_type == "Flatwing" else fuel_t),
            report_output_dir=report_output_dir,
            weight_df_flatwing=(weight_df_f if 'weight_df_f' in locals() else None),
            weight_df_tamarack=(weight_df_t if 'weight_df_t' in locals() else None),
            weight_df_single=(weight_df if 'weight_df' in locals() else None),
            modes_summary_df=(modes_summary_df if 'modes_summary_df' in locals() else None),
            fuel_cost_per_gal=fuel_cost_per_gal
        )
     
    # Display output file information (for all wing types)
    main.markdown("---")
    main.subheader('Output Files')
    
    output_files = []
    
    if wing_type == "Comparison":
        if "Tamarack" in mods_available and 'tamarack_output_file' in locals():
            output_files.append((f"{aircraft_model} Tamarack", tamarack_output_file))
        if "Flatwing" in mods_available and 'flatwing_output_file' in locals():
            output_files.append((f"{aircraft_model} Flatwing", flatwing_output_file))
    elif wing_type == "Tamarack" and 'tamarack_output_file' in locals():
        output_files.append((f"{aircraft_model} Tamarack", tamarack_output_file))
    elif wing_type == "Flatwing" and 'flatwing_output_file' in locals():
        output_files.append((f"{aircraft_model} Flatwing", flatwing_output_file))
    
    if output_files:
        main.write("**Time history data files have been created:**")
        for config_name, filepath in output_files:
            main.success(f" {config_name}: `{filepath}`")
            
        # Show directory information
        output_dir = os.path.dirname(output_files[0][1]) if output_files else "output"
        main.info(f" All files saved in: `{output_dir}`")
        main.markdown("**Downloads**")
        dl_cols = main.columns(min(3, max(1, len(output_files))))
        for i, (config_name, filepath) in enumerate(output_files):
            try:
                with open(filepath, "rb") as f:
                    data = f.read()
                filename = os.path.basename(filepath) or f"{config_name}.csv"
                with dl_cols[i % len(dl_cols)]:
                    main.download_button(
                        label=f"Download {config_name} CSV",
                        data=data,
                        file_name=filename,
                        mime="text/csv",
                    )
            except Exception:
                pass

        try:
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
                if os.path.isdir(output_dir):
                    for root, _, files in os.walk(output_dir):
                        for fn in files:
                            full_path = os.path.join(root, fn)
                            arc_name = os.path.relpath(full_path, start=output_dir)
                            zf.write(full_path, arcname=arc_name)
                else:
                    for _, filepath in output_files:
                        if os.path.isfile(filepath):
                            zf.write(filepath, arcname=os.path.basename(filepath))
            zip_name = f"{os.path.basename(output_dir) or 'outputs'}.zip"
            main.download_button(
                label="Download all outputs (zip)",
                data=buf.getvalue(),
                file_name=zip_name,
                mime="application/zip",
            )
        except Exception:
            pass
        main.write("*Files contain simulation parameters sampled every 5 seconds*")
