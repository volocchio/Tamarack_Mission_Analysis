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
if 'payload_input_flatwing' not in st.session_state:
    st.session_state.payload_input_flatwing = 0
if 'payload_input_tamarack' not in st.session_state:
    st.session_state.payload_input_tamarack = 0

# Sidebar inputs
with st.sidebar:
    st.header("Flight Parameters")
    
    # Aircraft model selection
    aircraft_types = ["CJ", "CJ1", "CJ1+", "M2", "CJ2", "CJ2+", "CJ3", "CJ3+"]
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
                  if "KSZT" in name.upper()), 0),
        format_func=lambda x: x,
        key="departure_airport"
    )
    
    arrival_airport = st.selectbox(
        "Arrival Airport",
        options=airport_display_names,
        index=next((i for i, name in enumerate(airport_display_names) 
                  if "KSJT" in name.upper()), 0),
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
            st.session_state.payload_input_tamarack = int(tamarack_max_payload)
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
        if 'payload_input_flatwing' not in st.session_state:
            st.session_state.payload_input_flatwing = 0
            
        # Always ensure the payload doesn't exceed max_payload_f
        current_payload = st.session_state.get('payload_input_flatwing', 0)
        payload_value = min(current_payload, int(max_payload_f))
        
        # Update payload if we're in Max Payload mode and either:
        # 1. The weight option just changed to Max Payload, or
        # 2. BOW was just modified while in Max Payload mode
        if weight_option == "Max Payload (Fill Payload to MZFW, Adjust Fuel to MRW)" and \
           (weight_option_changed or st.session_state.get("bow_changed", False)):
            payload_value = int(max_payload_f)
        
        # Update session state with the validated payload value
        st.session_state.payload_input_flatwing = payload_value
        
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
        if 'payload_input_tamarack' not in st.session_state:
            st.session_state.payload_input_tamarack = 0
            
        # Always ensure the payload doesn't exceed max_payload_t
        current_payload = st.session_state.get('payload_input_tamarack', 0)
        payload_value = min(current_payload, int(max_payload_t))
        
        # Update payload if we're in Max Payload mode and either:
        # 1. The weight option just changed to Max Payload, or
        # 2. BOW was just modified while in Max Payload mode
        if weight_option == "Max Payload (Fill Payload to MZFW, Adjust Fuel to MRW)" and \
           (weight_option_changed or st.session_state.get("bow_changed_t", False)):
            payload_value = int(max_payload_t)
        
        # Update session state with the validated payload value
        st.session_state.payload_input_tamarack = payload_value
        
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
    wing_type = st.radio("Wing Type", ["Flatwing", "Tamarack", "Comparison"], index=0, key="wing_type")
    if wing_type != "Comparison" and wing_type not in mods_available:
        st.error(f"Wing type '{wing_type}' is not available for aircraft model {aircraft_model}. Available options: {mods_available}")
        st.stop()

    # Takeoff flap selection
    flap_option = st.radio("Takeoff Flaps", ["Flap 0", "Flaps 15"], index=0)
    takeoff_flap = 1 if flap_option == "Flaps 15" else 0

    # Winds and temps source
    winds_temps_source = st.radio("Winds and Temps Aloft Source", 
                                ["No Wind", "Current Conditions", "Summer Average", "Winter Average"], 
                                index=1)  # Default to "Current Conditions"

    # ISA deviation
    isa_dev = int(st.number_input("ISA Deviation (C)", value=0.0, step=1.0))

    # V1 cut simulation
    v1_cut_enabled = st.checkbox("Enable V1 Cut Simulation (Single Engine)", value=False)

# Main content area for outputs
if st.button("Run Simulation"):
    try:
        # Get the selected display names
        dep_display_name = departure_airport
        arr_display_name = arrival_airport
        
        print(f"\n=== DEBUG ===")
        print(f"Selected departure: '{dep_display_name}'")
        print(f"Selected arrival: '{arr_display_name}'")
        print("\nFirst 5 available airports:")
        for name in airport_display_names[:5]:
            print(f"- '{name}'")
        
        # Check if the selected names are in our mapping
        print(f"\nDeparture in mapping:", dep_display_name in display_name_to_info)
        print(f"Arrival in mapping:", arr_display_name in display_name_to_info)
        
        # Try to find a match with different cases
        dep_matches = [name for name in display_name_to_info if name.upper() == dep_display_name.upper()]
        arr_matches = [name for name in display_name_to_info if name.upper() == arr_display_name.upper()]
        
        print(f"\nCase-insensitive matches for departure: {dep_matches}")
        print(f"Case-insensitive matches for arrival: {arr_matches}")
        
        # Get the airport info using case-insensitive matching
        dep_airport_info = next((display_name_to_info[name] for name in display_name_to_info 
                               if name.upper() == dep_display_name.upper()), None)
        arr_airport_info = next((display_name_to_info[name] for name in display_name_to_info 
                               if name.upper() == arr_display_name.upper()), None)
        
        if dep_airport_info is None:
            similar = [name for name in display_name_to_info 
                     if dep_display_name.upper() in name.upper()][:5]
            st.error(f"Departure airport '{dep_display_name}' not found. Similar: {similar}")
            st.stop()
            
        if arr_airport_info is None:
            similar = [name for name in display_name_to_info 
                     if arr_display_name.upper() in name.upper()][:5]
            st.error(f"Arrival airport '{arr_display_name}' not found. Similar: {similar}")
            st.stop()
        
        # Get the airport codes
        dep_airport_code = dep_airport_info["ident"]
        arr_airport_code = arr_airport_info["ident"]
        
        # Get coordinates and elevation
        dep_lat, dep_lon, elev_dep = dep_airport_info[["latitude_deg", "longitude_deg", "elevation_ft"]]
        arr_lat, arr_lon, elev_arr = arr_airport_info[["latitude_deg", "longitude_deg", "elevation_ft"]]
        
        # Calculate distance and bearing
        distance_nm, bearing_deg = haversine_with_bearing(dep_lat, dep_lon, arr_lat, arr_lon)

        # Display airport info
        st.header("Airport Information")
        
        # Create a custom CSS style for compact display
        compact_style = """
        <style>
        .compact-metric {
            font-size: 0.85rem;
            line-height: 1.2;
            margin-bottom: 0.5rem;
        }
        </style>
        """
        st.markdown(compact_style, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class="compact-metric">
                <div>Departure Elevation: {elev_dep:,} ft</div>
                <div>Arrival Elevation: {elev_arr:,} ft</div>
            </div>
            """.format(elev_dep=elev_dep, elev_arr=elev_arr), unsafe_allow_html=True)
            
            st.markdown("""
            <div class="compact-metric">
                <div>Distance: {distance_nm:.1f} NM</div>
                <div>Bearing: {bearing_deg:.1f}°</div>
            </div>
            """.format(distance_nm=distance_nm, bearing_deg=bearing_deg), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="compact-metric">
                <div>Departure DA: {da_dep:,} ft</div>
                <div>Arrival DA: {da_arr:,} ft</div>
            </div>
            """.format(da_dep=elev_dep + (120 * isa_dev), da_arr=elev_arr + (120 * isa_dev)), unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error calculating route: {str(e)}")
        st.stop()

    # Use user-adjusted values for simulation
    if wing_type == "Comparison":
        payload_f = payload_input_f
        fuel_f = fuel_input_f
        payload_t = payload_input_t
        fuel_t = fuel_input_t
    elif wing_type == "Flatwing":
        payload_f = payload_input_f
        fuel_f = fuel_input_f
        payload_t = 0
        fuel_t = 0
    elif wing_type == "Tamarack":
        payload_f = 0
        fuel_f = 0
        payload_t = payload_input_t
        fuel_t = fuel_input_t

    # Calculate current weights using user inputs
    if wing_type == "Comparison":
        # Flatwing weights
        zfw_f = bow_f + payload_input_f  # Zero Fuel Weight
        rw_f = zfw_f + fuel_input_f  # Ramp Weight
        tow_f = rw_f - taxi_fuel_f  # Takeoff Weight (after taxiing)
        mission_fuel_f = fuel_input_f - reserve_fuel_f - taxi_fuel_f
        
        # Tamarack weights - Always use values from config file
        tamarack_config = AIRCRAFT_CONFIG.get((aircraft_model, "Tamarack"))
        if tamarack_config:
            tamarack_mzfw = tamarack_config[19]  # MZFW is at index 19 in the config tuple
            tamarack_bow = tamarack_config[18]   # BOW is at index 18
            
            # Ensure payload doesn't exceed MZFW - BOW
            max_payload_t = max(0, tamarack_mzfw - tamarack_bow)
            if payload_input_t > max_payload_t:
                payload_input_t = max_payload_t
                # Instead of modifying session state directly, we'll use the adjusted value
                # and let the widget update on the next render
                
            zfw_t = tamarack_bow + payload_input_t  # Zero Fuel Weight
            rw_t = zfw_t + fuel_input_t  # Ramp Weight
            tow_t = rw_t - taxi_fuel_t  # Takeoff Weight (after taxiing)
            mission_fuel_t = fuel_input_t - reserve_fuel_t - taxi_fuel_t
        else:
            # Fallback to Flatwing calculation if Tamarack config not found (shouldn't happen)
            st.error("Tamarack configuration not found!")
            zfw_t = 0
            rw_t = 0
            tow_t = 0
            mission_fuel_t = 0
    elif wing_type == "Tamarack":
        # Set Flatwing weights to 0 for comparison
        zfw_f = 0
        rw_f = 0
        tow_f = 0
        mission_fuel_f = 0
        
        tamarack_config = AIRCRAFT_CONFIG.get((aircraft_model, "Tamarack"))
        if tamarack_config:
            tamarack_mzfw = tamarack_config[19]  # MZFW is at index 19 in the config tuple
            tamarack_bow = tamarack_config[18]   # BOW is at index 18
            
            # Ensure payload doesn't exceed MZFW - BOW
            max_payload_t = max(0, tamarack_mzfw - tamarack_bow)
            if payload_input_t > max_payload_t:
                payload_input_t = max_payload_t
                # Instead of modifying session state directly, we'll use the adjusted value
                # and let the widget update on the next render
                
            zfw_t = tamarack_bow + payload_input_t  # Zero Fuel Weight
            rw_t = zfw_t + fuel_input_t  # Ramp Weight
            tow_t = rw_t - taxi_fuel_t  # Takeoff Weight (after taxiing)
            mission_fuel_t = fuel_input_t - reserve_fuel_t - taxi_fuel_t
        else:
            st.error("Tamarack configuration not found!")
            zfw_t = 0
            rw_t = 0
            tow_t = 0
            mission_fuel_t = 0
    else:
        # For single configuration mode
        config = AIRCRAFT_CONFIG.get((aircraft_model, wing_type))
        if not config:
            st.error(f"No configuration found for {aircraft_model} with {wing_type} modification.")
            st.stop()
        
        try:
            # Unpack the first 35 values from the config tuple
            config_values = list(config)[:35]
            s, b, e, h, sweep_25c, sfc, engines_orig, thrust_mult, ceiling, CL0, CLA, cdo, dcdo_flap1, dcdo_flap2, \
                dcdo_flap3, dcdo_gear, mu_to, mu_lnd, bow, mzfw, mrw, mtow, max_fuel, \
                taxi_fuel_default, reserve_fuel_default, mmo, VMO, clmax, clmax_1, clmax_2, m_climb, \
                v_climb, roc_min, m_descent, v_descent = config_values
            
            max_payload = mzfw - bow
        except ValueError as e:
            st.error(f"Error extracting configuration values: {str(e)}")
            st.stop()

        if wing_type == "Flatwing":
            payload = payload_input_f
            fuel = fuel_input_f
            taxi_fuel = taxi_fuel_f
            reserve_fuel = reserve_fuel_f
            mission_fuel = fuel_input_f - reserve_fuel_f - taxi_fuel_f
            zfw = bow_f + payload_input_f  # Zero Fuel Weight
            rw = zfw + fuel_input_f  # Ramp Weight
            tow = rw - taxi_fuel_f  # Takeoff Weight (after taxiing)
            bow = bow_f  # Use the input BOW value
        else:  # Tamarack
            payload = payload_input_t
            fuel = fuel_input_t
            taxi_fuel = taxi_fuel_t
            reserve_fuel = reserve_fuel_t
            mission_fuel = fuel_input_t - reserve_fuel_t - taxi_fuel_t
            zfw = bow_t + payload_input_t  # Zero Fuel Weight
            rw = zfw + fuel_input_t  # Ramp Weight
            tow = rw - taxi_fuel_t  # Takeoff Weight (after taxiing)
            bow = bow_t  # Use the input BOW value

        weight_df = pd.DataFrame({
            'Component': [
                'BOW', 'Payload', 'Initial Fuel', 'Reserve Fuel', 'Taxi Fuel', 
                'Mission Fuel', 'ZFW', 'Ramp Weight', 'Takeoff Weight'
            ],
            'Weight (lb)': [
                f"{bow:,.0f}",
                f"{payload:,.0f}",
                f"{fuel:,.0f}",
                f"{reserve_fuel:,.0f}",
                f"{taxi_fuel:,.0f}",
                f"{mission_fuel:,.0f}",
                f"{zfw:,.0f}",
                f"{rw:,.0f}",
                f"{tow:,.0f}"
            ],
            'Max Weight (lb)': [
                "",  # BOW - no max
                f"{max_payload:,.0f}",  # Payload has max
                f"{max_fuel:,.0f}",  # Initial Fuel has max
                "",  # Reserve Fuel - no max
                "",  # Taxi Fuel - no max
                "",  # Mission Fuel - no max
                f"{mzfw:,.0f}",  # ZFW has max
                f"{mrw:,.0f}",  # Ramp Weight has max
                f"{mtow:,.0f}"  # Takeoff Weight has max
            ]
        })

        # Display weight status table before simulation
        st.markdown("---")
        st.subheader('Weight Status')
        def highlight_exceeded(row):
            weight = float(row['Weight (lb)'].replace(',', ''))
            max_weight = row['Max Weight (lb)']
            if max_weight and max_weight.strip():  # Only compare if max_weight is not empty
                try:
                    max_weight_val = float(max_weight.replace(',', ''))
                    if weight > max_weight_val:
                        return ['background-color: #ffcccc'] * len(row)
                except (ValueError, AttributeError):
                    pass
            return [''] * len(row)

        styled_df = weight_df.style.apply(highlight_exceeded, axis=1)
        st.table(styled_df)
        
        # Check for any exceeded weight limits
        def is_weight_exceeded(row):
            try:
                weight = float(row['Weight (lb)'].replace(',', ''))
                max_weight = row['Max Weight (lb)']
                if max_weight and max_weight.strip():
                    max_weight_val = float(max_weight.replace(',', ''))
                    return weight > max_weight_val
                return False
            except (ValueError, AttributeError):
                return False
        
        weight_exceeded = weight_df.apply(is_weight_exceeded, axis=1).any()
        
        if weight_exceeded:
            st.error("Weight limits exceeded! Please adjust the following:")
            for _, row in weight_df.iterrows():
                weight = float(row['Weight (lb)'].replace(',', ''))
                max_weight = float(row['Max Weight (lb)'].replace(',', ''))
                if weight > max_weight:
                    excess_amount = weight - max_weight
                    st.error(f"- {row['Component']}: Current {row['Weight (lb)']} lbs exceeds max {row['Max Weight (lb)']} lbs by {excess_amount:,.0f} lbs")
            st.stop()

    # Run simulation only if weights are valid
    tamarack_data = pd.DataFrame()
    tamarack_results = {}
    flatwing_data = pd.DataFrame()
    flatwing_results = {}

    if wing_type == "Comparison":
        if "Tamarack" in mods_available:
            tamarack_data, tamarack_results, dep_lat, dep_lon, arr_lat, arr_lon = run_simulation(
                dep_airport_code, arr_airport_code, aircraft_model, "Tamarack", takeoff_flap,
                payload_t, fuel_t, taxi_fuel_t, reserve_fuel_t, cruise_altitude_t,
                winds_temps_source, v1_cut_enabled)
            print(f"\n=== DEBUG: Tamarack Results from run_simulation ===")
            print(f"Tamarack Results keys: {list(tamarack_results.keys())}")
            print(f"Tamarack Takeoff V-Speeds: {tamarack_results.get('Takeoff V-Speeds')}")
            print(f"Tamarack Approach V-Speeds: {tamarack_results.get('Approach V-Speeds')}")
        if "Flatwing" in mods_available:
            flatwing_data, flatwing_results, dep_lat, dep_lon, arr_lat, arr_lon = run_simulation(
                dep_airport_code, arr_airport_code, aircraft_model, "Flatwing", takeoff_flap,
                payload_f, fuel_f, taxi_fuel_f, reserve_fuel_f, cruise_altitude_f,
                winds_temps_source, v1_cut_enabled)
            print(f"\n=== DEBUG: Flatwing Results from run_simulation ===")
            print(f"Flatwing Results keys: {list(flatwing_results.keys())}")
            print(f"Flatwing Takeoff V-Speeds: {flatwing_results.get('Takeoff V-Speeds')}")
            print(f"Flatwing Approach V-Speeds: {flatwing_results.get('Approach V-Speeds')}")
    elif wing_type == "Tamarack":
        tamarack_data, tamarack_results, dep_lat, dep_lon, arr_lat, arr_lon = run_simulation(
            dep_airport_code, arr_airport_code, aircraft_model, "Tamarack", takeoff_flap,
            payload_t, fuel_t, taxi_fuel_t, reserve_fuel_t, cruise_altitude_t,
            winds_temps_source, v1_cut_enabled)
        print(f"\n=== DEBUG: Tamarack Results from run_simulation ===")
        print(f"Tamarack Results keys: {list(tamarack_results.keys())}")
        print(f"Tamarack Takeoff V-Speeds: {tamarack_results.get('Takeoff V-Speeds')}")
        print(f"Tamarack Approach V-Speeds: {tamarack_results.get('Approach V-Speeds')}")
    elif wing_type == "Flatwing":
        flatwing_data, flatwing_results, dep_lat, dep_lon, arr_lat, arr_lon = run_simulation(
            dep_airport_code, arr_airport_code, aircraft_model, "Flatwing", takeoff_flap,
            payload_f, fuel_f, taxi_fuel_f, reserve_fuel_f, cruise_altitude_f,
            winds_temps_source, v1_cut_enabled)
        print(f"\n=== DEBUG: Flatwing Results from run_simulation ===")
        print(f"Flatwing Results keys: {list(flatwing_results.keys())}")
        print(f"Flatwing Takeoff V-Speeds: {flatwing_results.get('Takeoff V-Speeds')}")
        print(f"Flatwing Approach V-Speeds: {flatwing_results.get('Approach V-Speeds')}")

    if v1_cut_enabled:
        if not tamarack_data.empty:
            end_idx = tamarack_data[tamarack_data['Segment'] == 3].index[-1] if not tamarack_data[tamarack_data['Segment'] == 3].empty else 0
            tamarack_data = tamarack_data.iloc[:end_idx + 1]
        if not flatwing_data.empty:
            end_idx = flatwing_data[flatwing_data['Segment'] == 3].index[-1] if not flatwing_data[flatwing_data['Segment'] == 3].empty else 0
            flatwing_data = flatwing_data.iloc[:end_idx + 1]

    # Display simulation results
    st.markdown("---")
    st.header('Simulation Results')
    display_simulation_results(
        tamarack_data, tamarack_results,
        flatwing_data, flatwing_results,
        v1_cut_enabled,
        dep_lat, dep_lon, arr_lat, arr_lon,
        distance_nm, bearing_deg,
        winds_temps_source,
        cruise_altitude_f if wing_type == "Flatwing" else cruise_altitude_t,
        dep_airport_code,
        arr_airport_code,
        fuel_f if wing_type == "Flatwing" else fuel_t
    )