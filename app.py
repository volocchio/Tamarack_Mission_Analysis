import os
import streamlit as st
import pandas as pd
from aircraft_config import AIRCRAFT_CONFIG
from utils import load_airports
from simulation import run_simulation, haversine_with_bearing
from display import display_simulation_results

# --- Streamlit UI ---
st.title("Flight Simulation App")
st.markdown("""
This app simulates a flight between two airports using a specified aircraft model.
It calculates flight parameters such as altitude, speed, thrust, and drag over time,
and visualizes the flight profile with charts.
""")

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
    aircraft_types = ["CJ", "CJ1", "CJ1+", "M2", "CJ2", "CJ2+", "CJ3", "CJ3+", "C208", "C208B", "C208EX"]
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
        except FileNotFoundError:
            st.warning(f"Tamarack image not found: {image_path}")
        try:
            image_path = f"images/flatwing_{aircraft_model}.jpg"
            st.image(image_path, caption=f"Flatwing {aircraft_model}", use_container_width=True)
        except FileNotFoundError:
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
        else:
            tamarack_mzfw = mzfw
            tamarack_bow = bow
            
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
                  if name.startswith("KSAN")), 0),
        format_func=lambda x: x,
        key="arrival_airport"
    )

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
    winds_temps_source = st.radio("Winds and Temps Aloft Source", 
                                ["No Wind", "Current Conditions", "Summer Average", "Winter Average"], 
                                index=0)  # Default to "No Wind"

    # ISA deviation
    isa_dev = int(st.number_input("ISA Deviation (C)", value=0.0, step=1.0))

    # Performance tuning biases
    st.subheader("Performance Tuning (Biases)")
    st.markdown("Adjust SFC and thrust vs altitude to better match specific aircraft behavior.")
    col_bias1, col_bias2 = st.columns(2)
    with col_bias1:
        sfc_bias_low = st.slider(
            "Low Alt SFC Bias (%)",
            min_value=-20,
            max_value=20,
            value=0,
            step=1,
            help="Percent bias applied to SFC near sea level (negative = lower SFC, positive = higher SFC).",
        )
        thrust_bias_low = st.slider(
            "Low Alt Thrust Bias (%)",
            min_value=-20,
            max_value=20,
            value=0,
            step=1,
            help="Percent bias applied to thrust near sea level (negative = less thrust, positive = more thrust).",
        )
    with col_bias2:
        sfc_bias_high = st.slider(
            "High Alt SFC Bias (%)",
            min_value=-20,
            max_value=20,
            value=0,
            step=1,
            help="Percent bias applied to SFC at high altitude cruise (negative = lower SFC, positive = higher SFC).",
        )
        thrust_bias_high = st.slider(
            "High Alt Thrust Bias (%)",
            min_value=-20,
            max_value=20,
            value=0,
            step=1,
            help="Percent bias applied to thrust at high altitude cruise (negative = less thrust, positive = more thrust).",
        )

    # Cruise mode selection
    cruise_mode = st.radio(
        "Cruise Mode",
        ["MCT (Max Thrust)", "Max Range", "Max Endurance"],
        index=0,
        format_func=lambda x: {"Max Range": "Max Range speed (LRC)", "Max Endurance": "Max Endurance Speed"}.get(x, x),
        help="Above 10,000 ft: set cruise speed by objective. For MCT, hold max thrust. For Max Range/Endurance, target optimal CL/CD with V >= 1.2*Vs."
    )
    run_all_modes = st.checkbox("Run All Modes", value=False, help="Run MCT, LRC, and Max Endurance; show detailed results for the selected mode above.")

    # V1 cut simulation
    v1_cut_enabled = st.checkbox("Enable V1 Cut Simulation (Single Engine)", value=False)

    # Output file option
    write_output_file = st.checkbox("Write Output CSV File", value=True)
    fuel_cost_per_gal = st.number_input("Fuel Cost ($/gal)", min_value=0.0, value=5.0, step=0.1)

# Main content area for outputs
if st.button("Run Simulation"):
    # Prepare airport/route variables used later
    try:
        dep_display_name = departure_airport
        arr_display_name = arrival_airport
        dep_airport_info = next((display_name_to_info[name] for name in display_name_to_info 
                           if name.upper() == dep_display_name.upper()), None)
        arr_airport_info = next((display_name_to_info[name] for name in display_name_to_info 
                           if name.upper() == arr_display_name.upper()), None)
        if dep_airport_info is None or arr_airport_info is None:
            dep_airport_code = ""
            arr_airport_code = ""
            dep_lat = dep_lon = arr_lat = arr_lon = 0.0
            distance_nm = 0.0
            bearing_deg = 0.0
        else:
            dep_airport_code = dep_airport_info["ident"]
            arr_airport_code = arr_airport_info["ident"]
            dep_lat, dep_lon, elev_dep = dep_airport_info[["latitude_deg", "longitude_deg", "elevation_ft"]]
            arr_lat, arr_lon, elev_arr = arr_airport_info[["latitude_deg", "longitude_deg", "elevation_ft"]]
            distance_nm, bearing_deg = haversine_with_bearing(dep_lat, dep_lon, arr_lat, arr_lon)
    except Exception:
        dep_airport_code = ""
        arr_airport_code = ""
        dep_lat = dep_lon = arr_lat = arr_lon = 0.0
        distance_nm = 0.0
        bearing_deg = 0.0

    # Run simulation only if weights are valid
    tamarack_data = pd.DataFrame()
    tamarack_results = {}
    flatwing_data = pd.DataFrame()
    flatwing_results = {}
    modes_summary_df = None

    # Selected mode for detailed results
    chosen_mode = cruise_mode

    # Prepare per-wing inputs used by simulations
    if wing_type == "Comparison":
        payload_f = payload_input_f
        fuel_f = fuel_input_f
        payload_t = payload_input_t
        fuel_t = fuel_input_t
    elif wing_type == "Flatwing":
        payload_f = payload_input_f
        fuel_f = fuel_input_f
    elif wing_type == "Tamarack":
        payload_t = payload_input_t
        fuel_t = fuel_input_t

    if run_all_modes:
        modes_to_run = ["MCT (Max Thrust)", "Max Range", "Max Endurance"]
        # Preserve order but put chosen mode first for summary display
        ordered_modes = [m for m in modes_to_run if m == chosen_mode] + [m for m in modes_to_run if m != chosen_mode]
        results_by_mode = {}

        for mode in modes_to_run:
            res = {}
            if wing_type == "Comparison":
                if "Tamarack" in mods_available:
                    t_data, t_results, dep_lat, dep_lon, arr_lat, arr_lon, t_out = run_simulation(
                        dep_airport_code, arr_airport_code, aircraft_model, "Tamarack", takeoff_flap,
                        payload_t, fuel_t, taxi_fuel_t, reserve_fuel_t, cruise_altitude_t,
                        winds_temps_source, v1_cut_enabled, write_output_file, cruise_mode=mode,
                        sfc_bias_low=sfc_bias_low, sfc_bias_high=sfc_bias_high,
                        thrust_bias_low=thrust_bias_low, thrust_bias_high=thrust_bias_high)
                    res["Tamarack"] = (t_data, t_results, t_out)
                if "Flatwing" in mods_available:
                    f_data, f_results, dep_lat, dep_lon, arr_lat, arr_lon, f_out = run_simulation(
                        dep_airport_code, arr_airport_code, aircraft_model, "Flatwing", takeoff_flap,
                        payload_f, fuel_f, taxi_fuel_f, reserve_fuel_f, cruise_altitude_f,
                        winds_temps_source, v1_cut_enabled, write_output_file, cruise_mode=mode,
                        sfc_bias_low=sfc_bias_low, sfc_bias_high=sfc_bias_high,
                        thrust_bias_low=thrust_bias_low, thrust_bias_high=thrust_bias_high)
                    res["Flatwing"] = (f_data, f_results, f_out)
            elif wing_type == "Tamarack":
                t_data, t_results, dep_lat, dep_lon, arr_lat, arr_lon, t_out = run_simulation(
                    dep_airport_code, arr_airport_code, aircraft_model, "Tamarack", takeoff_flap,
                    payload_t, fuel_t, taxi_fuel_t, reserve_fuel_t, cruise_altitude_t,
                    winds_temps_source, v1_cut_enabled, write_output_file, cruise_mode=mode,
                    sfc_bias_low=sfc_bias_low, sfc_bias_high=sfc_bias_high,
                    thrust_bias_low=thrust_bias_low, thrust_bias_high=thrust_bias_high)
                res["Tamarack"] = (t_data, t_results, t_out)
            elif wing_type == "Flatwing":
                f_data, f_results, dep_lat, dep_lon, arr_lat, arr_lon, f_out = run_simulation(
                    dep_airport_code, arr_airport_code, aircraft_model, "Flatwing", takeoff_flap,
                    payload_f, fuel_f, taxi_fuel_f, reserve_fuel_f, cruise_altitude_f,
                    winds_temps_source, v1_cut_enabled, write_output_file, cruise_mode=mode,
                    sfc_bias_low=sfc_bias_low, sfc_bias_high=sfc_bias_high,
                    thrust_bias_low=thrust_bias_low, thrust_bias_high=thrust_bias_high)
                res["Flatwing"] = (f_data, f_results, f_out)
            results_by_mode[mode] = res

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
        if wing_type == "Comparison":
            if "Tamarack" in mods_available:
                tamarack_data, tamarack_results, dep_lat, dep_lon, arr_lat, arr_lon, tamarack_output_file = run_simulation(
                    dep_airport_code, arr_airport_code, aircraft_model, "Tamarack", takeoff_flap,
                    payload_t, fuel_t, taxi_fuel_t, reserve_fuel_t, cruise_altitude_t,
                    winds_temps_source, v1_cut_enabled, write_output_file, cruise_mode=cruise_mode,
                    sfc_bias_low=sfc_bias_low, sfc_bias_high=sfc_bias_high,
                    thrust_bias_low=thrust_bias_low, thrust_bias_high=thrust_bias_high)
            if "Flatwing" in mods_available:
                flatwing_data, flatwing_results, dep_lat, dep_lon, arr_lat, arr_lon, flatwing_output_file = run_simulation(
                    dep_airport_code, arr_airport_code, aircraft_model, "Flatwing", takeoff_flap,
                    payload_f, fuel_f, taxi_fuel_f, reserve_fuel_f, cruise_altitude_f,
                    winds_temps_source, v1_cut_enabled, write_output_file, cruise_mode=cruise_mode,
                    sfc_bias_low=sfc_bias_low, sfc_bias_high=sfc_bias_high,
                    thrust_bias_low=thrust_bias_low, thrust_bias_high=thrust_bias_high)
        elif wing_type == "Tamarack":
            tamarack_data, tamarack_results, dep_lat, dep_lon, arr_lat, arr_lon, tamarack_output_file = run_simulation(
                dep_airport_code, arr_airport_code, aircraft_model, "Tamarack", takeoff_flap,
                payload_t, fuel_t, taxi_fuel_t, reserve_fuel_t, cruise_altitude_t,
                winds_temps_source, v1_cut_enabled, write_output_file, cruise_mode=cruise_mode,
                sfc_bias_low=sfc_bias_low, sfc_bias_high=sfc_bias_high,
                thrust_bias_low=thrust_bias_low, thrust_bias_high=thrust_bias_high)
        elif wing_type == "Flatwing":
            flatwing_data, flatwing_results, dep_lat, dep_lon, arr_lat, arr_lon, flatwing_output_file = run_simulation(
                dep_airport_code, arr_airport_code, aircraft_model, "Flatwing", takeoff_flap,
                payload_f, fuel_f, taxi_fuel_f, reserve_fuel_f, cruise_altitude_f,
                winds_temps_source, v1_cut_enabled, write_output_file, cruise_mode=cruise_mode,
                sfc_bias_low=sfc_bias_low, sfc_bias_high=sfc_bias_high,
                thrust_bias_low=thrust_bias_low, thrust_bias_high=thrust_bias_high)

    if v1_cut_enabled:
        if not tamarack_data.empty:
            end_idx = tamarack_data[tamarack_data['Segment'] == 3].index[-1] if not tamarack_data[tamarack_data['Segment'] == 3].empty else 0

    st.markdown("---")
    st.header('Simulation Results')
    # Determine output folder for report (same as CSV)
    report_output_dir = None
    if 'tamarack_output_file' in locals():
        report_output_dir = os.path.dirname(tamarack_output_file)
    elif 'flatwing_output_file' in locals():
        report_output_dir = os.path.dirname(flatwing_output_file)

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
    st.markdown("---")
    st.subheader('Output Files')
    
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
        st.write("**Time history data files have been created:**")
        for config_name, filepath in output_files:
            st.success(f" {config_name}: `{filepath}`")
            
        # Show directory information
        output_dir = os.path.dirname(output_files[0][1]) if output_files else "output"
        st.info(f" All files saved in: `{output_dir}`")
        st.write("*Files contain simulation parameters sampled every 5 seconds*")