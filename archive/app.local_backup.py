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
airport_display_names = airports_df["display_name"].tolist()

# Initialize session state
if 'initial_values' not in st.session_state:
    st.session_state.initial_values = {}

# Add this near the top of your file, after imports
if 'bow_flatwing' not in st.session_state:
    st.session_state.bow_flatwing = None
if 'bow_tamarack' not in st.session_state:
    st.session_state.bow_tamarack = None

def update_payload_flatwing():
    if 'bow_flatwing' in st.session_state and 'payload_input_flatwing' in st.session_state:
        bow = st.session_state.bow_flatwing
        current_mzfw = tamarack_mzfw if wing_type == "Tamarack" else flatwing_mzfw
        max_payload = max(0, current_mzfw - bow)
        
        # Update payload if it exceeds the new max
        if st.session_state.payload_input_flatwing > max_payload:
            st.session_state.payload_input_flatwing = max_payload
        
        # Update the help text to show current MZFW and max payload
        st.session_state.payload_help_flatwing = f"Maximum payload: {int(max_payload):,} lb (MZFW: {int(current_mzfw):,} - BOW)"

def update_payload_tamarack():
    if 'bow_tamarack' in st.session_state and 'payload_input_tamarack' in st.session_state:
        bow = st.session_state.bow_tamarack
        current_mzfw = tamarack_mzfw  # Tamarack always uses its own MZFW
        max_payload = max(0, current_mzfw - bow)
        
        # Update payload if it exceeds the new max
        if st.session_state.payload_input_tamarack > max_payload:
            st.session_state.payload_input_tamarack = max_payload
        
        # Update the help text to show current MZFW and max payload
        st.session_state.payload_help_tamarack = f"Maximum payload: {int(max_payload):,} lb (MZFW: {int(current_mzfw):,} - BOW)"

# Sidebar inputs
with st.sidebar:
    st.header("Flight Parameters")
    
    # Aircraft model selection
    aircraft_types = ["CJ", "CJ1", "CJ1+", "M2", "CJ2", "CJ2+", "CJ3", "CJ3+"]
    aircraft_model = st.selectbox("Aircraft Model", aircraft_types, 
                                 index=aircraft_types.index("CJ1") if "CJ1" in aircraft_types else 0,
                                 key="aircraft_model")

    # Load aircraft config first
    mods_available = [m for (a, m) in AIRCRAFT_CONFIG if a == aircraft_model]
    if not mods_available:
        st.error(f"No modifications available for aircraft model {aircraft_model}.")
        st.stop()

    # Wing type selection
    wing_type = st.radio("Wing Type", ["Flatwing", "Tamarack", "Comparison"], 
                        index=0, key="wing_type")
    
    # Get the appropriate config based on wing type
    config = AIRCRAFT_CONFIG.get((aircraft_model, wing_type))
    if not config:
        st.error(f"No configuration found for {aircraft_model} with {wing_type} modification.")
        st.stop()

    # Extract values from the selected config
    try:
        # Print the config to debug
        st.write("Debug - Config values:", config)
        
        # Ensure config has the right number of elements
        if len(config) != 35:  # 35 elements in the tuple (0-34)
            st.error(f"Invalid configuration length: expected 35 elements, got {len(config)}")
            st.stop()
            
        # Unpack with explicit indices to ensure correct mapping
        s = config[0]           # Wing area (ft^2)
        b = config[1]           # Wing span (ft)
        e = config[2]           # Oswald efficiency factor
        h = config[3]           # Winglet height (ft)
        sweep_25c = config[4]   # Wing sweep at 25% chord (degrees)
        sfc = config[5]         # Specific Fuel Consumption (lb/hr/lb)
        engines_orig = config[6] # Number of engines
        thrust_mult = config[7]  # Thrust multiplier
        ceiling = config[8]      # Service ceiling (ft)
        CL0 = config[9]         # Zero-lift coefficient
        CLA = config[10]        # Lift curve slope (1/rad)
        cdo = config[11]        # Zero-lift drag coefficient
        dcdo_flap1 = config[12] # Drag coefficient increment for takeoff flaps 15
        dcdo_flap2 = config[13] # Drag coefficient increment for takeoff flaps 30
        dcdo_flap3 = config[14] # Drag coefficient increment for ground flaps and spoilers 40
        dcdo_gear = config[15]  # Drag coefficient increment for landing gear
        mu_to = config[16]      # Rolling friction coefficient during takeoff
        mu_lnd = config[17]     # Rolling friction coefficient during landing
        bow = config[18]        # Basic Operating Weight (lb)
        mzfw = config[19]       # Maximum Zero Fuel Weight (lb)
        mrw = config[20]        # Maximum Ramp Weight (lb)
        mtow = config[21]       # Maximum Takeoff Weight (lb)
        max_fuel = config[22]   # Maximum fuel capacity (lb)
        taxi_fuel_default = config[23]  # Taxi fuel allowance (lb)
        reserve_fuel_default = config[24]  # Reserve fuel requirement (lb)
        mmo = config[25]        # Maximum Mach number
        VMO = config[26]        # Maximum Operating Speed (kts)
        clmax = config[27]      # Maximum lift coefficient (clean)
        clmax_1 = config[28]    # Maximum lift coefficient (flaps 15)
        clmax_2 = config[29]    # Maximum lift coefficient (flaps 40)
        m_climb = config[30]    # Mach number for climb
        v_climb = config[31]    # Climb speed (kts)
        roc_min = config[32]    # Minimum rate of climb (ft/min)
        m_descent = config[33]  # Mach number for descent
        v_descent = config[34]  # Descent speed (kts)
            
        # Debug values
        st.write(f"Debug - bow: {bow}, mzfw: {mzfw}")
        st.write(f"Debug - bow type: {type(bow)}, mzfw type: {type(mzfw)}")
            
        # Ensure we have valid values
        if mzfw is None or bow is None:
            st.error(f"Invalid configuration: MZFW ({mzfw}) or BOW ({bow}) is not set")
            st.stop()
            
    except Exception as e:
        st.error(f"Error extracting configuration values: {str(e)}")
        st.error(f"Config length: {len(config) if hasattr(config, '__len__') else 'N/A'}, expected: 35")
        st.stop()

    # Weight mode selection
    weight_option = st.radio("Weight Configuration", [
        "Manual Input",
        "Max Fuel (Fill Tanks, Adjust Payload to MRW)",
        "Max Payload (Fill Payload to MZFW, Adjust Fuel to MRW)"
    ], index=0, key="weight_option")

    # Get initial values based on weight option
    if weight_option == "Max Fuel (Fill Tanks, Adjust Payload to MRW)":
        initial_fuel = float(max_fuel)
        max_payload = mzfw - bow
        initial_payload = float(min(max_payload, mrw - (bow + max_fuel)))
        rw = bow + initial_payload + initial_fuel
        tow = rw - taxi_fuel_default
        if tow > mtow:
            initial_fuel = mtow - (bow + initial_payload) + taxi_fuel_default
            if initial_fuel < 0:
                initial_fuel = 0
                initial_payload = mtow - bow + taxi_fuel_default
    elif weight_option == "Max Payload (Fill Payload to MZFW, Adjust Fuel to MRW)":
        max_payload = mzfw - bow
        initial_payload = float(max_payload)
        initial_fuel = float(min(max_fuel, mrw - (bow + max_payload)))
        if initial_fuel == max_fuel:
            initial_payload = float(min(max_payload, mrw - (bow + max_fuel)))
    else:
        # Set initial values based on aircraft model
        initial_fuel = 3440.0  # Default fuel
        initial_payload = 0.0
        max_payload = mzfw - bow

    # Store initial values in session state
    st.session_state.initial_values = {
        'payload': int(initial_payload),
        'fuel': int(initial_fuel),
        'taxi_fuel': int(taxi_fuel_default),
        'reserve_fuel': int(reserve_fuel_default),
        'cruise_altitude': int(ceiling)
    }

    # Initialize session state for payload values if they don't exist
    if 'payload_input_flatwing' not in st.session_state:
        st.session_state.payload_input_flatwing = 0.0
    if 'payload_input_tamarack' not in st.session_state:
        st.session_state.payload_input_tamarack = 0.0

    # Weight inputs - Flatwing
    st.subheader('Flatwing Weight Adjustment')

    # Create three columns for the inputs
    col1, col2, col3 = st.columns(3)

    with col1:
        # First column - BOW and Payload
        # Create the BOW input with a unique key
        bow_f = st.number_input(
            "BOW (lb)",
            min_value=0,
            max_value=int(mzfw) if mzfw is not None else 10000,
            value=int(bow) if bow is not None else 0,
            step=100,
            help="Basic Operating Weight (Empty Weight + pilot)",
            key="bow_flatwing"
        )
        
        # Calculate max payload based on BOW and MZFW
        if mzfw is not None and bow_f is not None:
            try:
                max_payload_f = max(0, float(mzfw) - float(bow_f))
                
                # Get current payload from session state or use 0
                current_payload = st.session_state.get('payload_input_flatwing', 0)
                if current_payload is None:
                    current_payload = 0
                current_payload = min(float(current_payload), float(max_payload_f))
                
                payload_input_f = st.number_input(
                    "Payload (lb)",
                    min_value=0,
                    max_value=int(max_payload_f),
                    value=int(current_payload),
                    step=100,
                    help=f"Maximum payload: {int(max_payload_f):,} lb (MZFW: {int(mzfw):,} - BOW)",
                    key="payload_input_flatwing"
                )
                
            except (ValueError, TypeError) as e:
                st.error(f"Error calculating payload: {str(e)}")
        else:
            st.error("Error: MZFW or BOW not properly configured")

    with col2:
        # Second column - Fuel inputs
        fuel_input_f = st.number_input(
            "Fuel (lb)",
            min_value=0,
            max_value=float(max_fuel) if max_fuel is not None else 10000.0,
            value=float(st.session_state.initial_values.get('fuel', 3440.0)),
            step=100.0,
            help=f"Maximum fuel: {int(max_fuel):,} lb",
            key="fuel_input_flatwing"
        )

        reserve_fuel_f = st.number_input(
            "Reserve Fuel (lb)",
            min_value=0.0,
            max_value=10000.0,
            value=float(st.session_state.initial_values.get('reserve_fuel', 600.0)),
            step=10.0,
            key="reserve_fuel_flatwing"
        )
    
    with col3:
        # Third column - Taxi and Altitude
        taxi_fuel_f = st.number_input(
            "Taxi Fuel (lb)",
            min_value=0.0,
            max_value=1000.0,
            value=float(st.session_state.initial_values.get('taxi_fuel', 100.0)),
            step=10.0,
            key="taxi_fuel_flatwing"
        )
        
        cruise_altitude_f = st.number_input(
            "Cruise Altitude Goal (ft)",
            min_value=0.0,
            max_value=float(ceiling) if ceiling is not None else 50000.0,
            value=float(st.session_state.initial_values.get('cruise_altitude', 41000.0)),
            step=1000.0,
            key="cruise_altitude_input_flatwing"
        )

    # Weight inputs - Tamarack
    st.subheader('Tamarack Weight Adjustment')

    # Create three columns for the inputs
    col4, col5, col6 = st.columns(3)

    with col4:
        # First column - BOW and Payload
        # Create the BOW input with a unique key
        bow_t = st.number_input(
            "BOW (lb)",
            min_value=0,
            max_value=int(mzfw) if mzfw is not None else 10000,
            value=int(bow) if bow is not None else 0,
            step=100,
            help="Basic Operating Weight (Empty Weight + pilot)",
            key="bow_tamarack"
        )
        
        # Calculate max payload based on BOW and MZFW
        if mzfw is not None and bow_t is not None:
            try:
                max_payload_t = max(0, float(mzfw) - float(bow_t))
                
                # Get current payload from session state or use 0
                current_payload = st.session_state.get('payload_input_tamarack', 0)
                if current_payload is None:
                    current_payload = 0
                current_payload = min(float(current_payload), float(max_payload_t))
                
                payload_input_t = st.number_input(
                    "Payload (lb)",
                    min_value=0,
                    max_value=int(max_payload_t),
                    value=int(current_payload),
                    step=100,
                    help=f"Maximum payload: {int(max_payload_t):,} lb (MZFW: {int(mzfw):,} - BOW)",
                    key="payload_input_tamarack"
                )
                
            except (ValueError, TypeError) as e:
                st.error(f"Error calculating payload: {str(e)}")
        else:
            st.error("Error: MZFW or BOW not properly configured")

    with col5:
        # Second column - Fuel inputs
        fuel_input_t = st.number_input(
            "Fuel (lb)",
            min_value=0.0,
            max_value=float(max_fuel) if max_fuel is not None else 10000.0,
            value=float(st.session_state.initial_values.get('fuel', 3440.0)),
            step=100.0,
            help=f"Maximum fuel: {int(max_fuel):,} lb",
            key="fuel_input_tamarack"
        )

        reserve_fuel_t = st.number_input(
            "Reserve Fuel (lb)",
            min_value=0.0,
            max_value=10000.0,
            value=float(st.session_state.initial_values.get('reserve_fuel', 600.0)),
            step=10.0,
            key="reserve_fuel_tamarack"
        )
    
    with col6:
        # Third column - Taxi and Altitude
        taxi_fuel_t = st.number_input(
            "Taxi Fuel (lb)",
            min_value=0.0,
            max_value=1000.0,
            value=float(st.session_state.initial_values.get('taxi_fuel', 100.0)),
            step=10.0,
            key="taxi_fuel_tamarack"
        )
        
        cruise_altitude_t = st.number_input(
            "Cruise Altitude Goal (ft)",
            min_value=0.0,
            max_value=float(ceiling) if ceiling is not None else 50000.0,
            value=float(st.session_state.initial_values.get('cruise_altitude', 41000.0)),
            step=1000.0,
            key="cruise_altitude_input_tamarack"
        )

    # Takeoff flap selection
    flap_option = st.radio("Takeoff Flaps", ["Flap 0", "Flaps 15"], index=0)
    takeoff_flap = 1 if flap_option == "Flaps 15" else 0

    # Winds and temps source
    winds_temps_source = st.radio("Winds and Temps Aloft Source", ["Current Conditions", "Summer Average", "Winter Average"], index=0)

    # ISA deviation
    isa_dev = int(st.number_input("ISA Deviation (C)", value=0.0, step=1.0))

    # V1 cut simulation
    v1_cut_enabled = st.checkbox("Enable V1 Cut Simulation (Single Engine)", value=False)

# Main content area for outputs
if st.button("Run Simulation"):
    # Get airport codes from display names
    try:
        # Get the selected display names
        dep_display_name = departure_airport
        arr_display_name = arrival_airport
        
        # Get airport info using exact display name match
        dep_airport_info = airports_df[airports_df["display_name"] == dep_display_name]
        arr_airport_info = airports_df[airports_df["display_name"] == arr_display_name]
        
        if dep_airport_info.empty:
            st.error(f"Departure airport '{dep_display_name}' not found in database")
            st.stop()
        if arr_airport_info.empty:
            st.error(f"Arrival airport '{arr_display_name}' not found in database")
            st.stop()
            
        # Get the airport codes from the filtered DataFrames
        dep_airport_code = dep_airport_info["ident"].iloc[0]
        arr_airport_code = arr_airport_info["ident"].iloc[0]
        
        # Get airport info again using codes
        dep_info = airports_df[airports_df["ident"] == dep_airport_code]
        arr_info = airports_df[airports_df["ident"] == arr_airport_code]
        
        if dep_info.empty or arr_info.empty:
            st.error("Invalid airport code(s). Please check selection.")
            st.stop()

        # Get coordinates and elevation
        dep_lat, dep_lon, elev_dep = dep_info.iloc[0][["latitude_deg", "longitude_deg", "elevation_ft"]]
        arr_lat, arr_lon, elev_arr = arr_info.iloc[0][["latitude_deg", "longitude_deg", "elevation_ft"]]
        
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
        
        # Tamarack weights
        zfw_t = bow_t + payload_input_t  # Zero Fuel Weight
        rw_t = zfw_t + fuel_input_t  # Ramp Weight
        tow_t = rw_t - taxi_fuel_t  # Takeoff Weight (after taxiing)
        mission_fuel_t = fuel_input_t - reserve_fuel_t - taxi_fuel_t
    elif wing_type == "Flatwing":
        zfw_f = bow_f + payload_input_f  # Zero Fuel Weight
        rw_f = zfw_f + fuel_input_f  # Ramp Weight
        tow_f = rw_f - taxi_fuel_f  # Takeoff Weight (after taxiing)
        mission_fuel_f = fuel_input_f - reserve_fuel_f - taxi_fuel_f
        
        # Set Tamarack weights to 0 for comparison
        zfw_t = 0
        rw_t = 0
        tow_t = 0
        mission_fuel_t = 0
    else:  # Tamarack
        # Set Flatwing weights to 0 for comparison
        zfw_f = 0
        rw_f = 0
        tow_f = 0
        mission_fuel_f = 0
        
        zfw_t = bow_t + payload_input_t  # Zero Fuel Weight
        rw_t = zfw_t + fuel_input_t  # Ramp Weight
        tow_t = rw_t - taxi_fuel_t  # Takeoff Weight (after taxiing)
        mission_fuel_t = fuel_input_t - reserve_fuel_t - taxi_fuel_t

    # Get configurations based on wing type
    if wing_type == "Comparison":
        # Get Flatwing configuration
        flatwing_config = AIRCRAFT_CONFIG.get((aircraft_model, "Flatwing"))
        if not flatwing_config:
            st.error(f"No Flatwing configuration found for {aircraft_model}")
            st.stop()
        
        # Get Tamarack configuration
        tamarack_config = AIRCRAFT_CONFIG.get((aircraft_model, "Tamarack"))
        if not tamarack_config:
            st.error(f"No Tamarack configuration found for {aircraft_model}")
            st.stop()

        # Create weight status dataframe for Flatwing
        try:
            s_f, b_f, e_f, h_f, sweep_25c_f, sfc_f, engines_orig_f, thrust_mult_f, ceiling_f, CL0_f, CLA_f, cdo_f, dcdo_flap1_f, dcdo_flap2_f, \
                dcdo_flap3_f, dcdo_gear_f, mu_to_f, mu_lnd_f, bow_f, mzfw_f, mrw_f, mtow_f, max_fuel_f, \
                taxi_fuel_default_f, reserve_fuel_default_f, mmo_f, VMO_f, clmax_f, clmax_1_f, clmax_2_f, m_climb_f, \
                v_climb_f, roc_min_f, m_descent_f, v_descent_f = flatwing_config
            max_payload_f = mzfw_f - bow_f
        except ValueError as e:
            st.error(f"Error extracting Flatwing configuration values: {str(e)}")
            st.stop()

        # Create weight status dataframe for Tamarack
        try:
            s_t, b_t, e_t, h_t, sweep_25c_t, sfc_t, engines_orig_t, thrust_mult_t, ceiling_t, CL0_t, CLA_t, cdo_t, dcdo_flap1_t, dcdo_flap2_t, \
                dcdo_flap3_t, dcdo_gear_t, mu_to_t, mu_lnd_t, bow_t, mzfw_t, mrw_t, mtow_t, max_fuel_t, \
                taxi_fuel_default_t, reserve_fuel_default_t, mmo_t, VMO_t, clmax_t, clmax_1_t, clmax_2_t, m_climb_t, \
                v_climb_t, roc_min_t, m_descent_t, v_descent_t = tamarack_config
            max_payload_t = mzfw_t - bow_t
        except ValueError as e:
            st.error(f"Error extracting Tamarack configuration values: {str(e)}")
            st.stop()

        # Create weight status dataframe for Flatwing
        weight_df = pd.DataFrame({
            'Component': [
                'BOW', 'Payload', 'Initial Fuel', 'Reserve Fuel', 'Taxi Fuel', 
                'Mission Fuel', 'ZFW', 'Ramp Weight', 'Takeoff Weight'
            ],
            'Weight (lb)': [
                f"{bow_f:,.0f}",
                f"{payload_input_f:,.0f}",
                f"{fuel_input_f:,.0f}",
                f"{reserve_fuel_f:,.0f}",
                f"{taxi_fuel_f:,.0f}",
                f"{mission_fuel_f:,.0f}",
                f"{zfw_f:,.0f}",
                f"{rw_f:,.0f}",
                f"{tow_f:,.0f}"
            ],
            'Max Weight (lb)': [
                "",  # BOW - no max
                f"{max_payload_f:,.0f}",  # Payload has max
                f"{max_fuel_f:,.0f}",  # Initial Fuel has max
                "",  # Reserve Fuel - no max
                "",  # Taxi Fuel - no max
                "",  # Mission Fuel - no max
                f"{mzfw_f:,.0f}",  # ZFW has max
                f"{mrw_f:,.0f}",  # Ramp Weight has max
                f"{mtow_f:,.0f}"  # Takeoff Weight has max
            ]
        })

        # Create weight status dataframe for Tamarack
        weight_df_tamarack = pd.DataFrame({
            'Component': [
                'BOW', 'Payload', 'Initial Fuel', 'Reserve Fuel', 'Taxi Fuel', 
                'Mission Fuel', 'ZFW', 'Ramp Weight', 'Takeoff Weight'
            ],
            'Weight (lb)': [
                f"{bow_t:,.0f}",
                f"{payload_input_t:,.0f}",
                f"{fuel_input_t:,.0f}",
                f"{reserve_fuel_t:,.0f}",
                f"{taxi_fuel_t:,.0f}",
                f"{mission_fuel_t:,.0f}",
                f"{zfw_t:,.0f}",
                f"{rw_t:,.0f}",
                f"{tow_t:,.0f}"
            ],
            'Max Weight (lb)': [
                "",  # BOW - no max
                f"{max_payload_t:,.0f}",  # Payload has max
                f"{max_fuel_t:,.0f}",  # Initial Fuel has max
                "",  # Reserve Fuel - no max
                "",  # Taxi Fuel - no max
                "",  # Mission Fuel - no max
                f"{mzfw_t:,.0f}",  # ZFW has max
                f"{mrw_t:,.0f}",  # Ramp Weight has max
                f"{mtow_t:,.0f}"  # Takeoff Weight has max
            ]
        })

        # Display weight status tables before simulation
        st.markdown("---")
        st.subheader('Weight Status')
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Flatwing Weight Status")
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
        
        with col2:
            st.subheader("Tamarack Weight Status")
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

            styled_df_tamarack = weight_df_tamarack.style.apply(highlight_exceeded, axis=1)
            st.table(styled_df_tamarack)
            
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
            
            weight_exceeded_t = weight_df_tamarack.apply(is_weight_exceeded, axis=1).any()
            
            if weight_exceeded_t:
                st.error("Weight limits exceeded! Please adjust the following:")
                for _, row in weight_df_tamarack.iterrows():
                    weight = float(row['Weight (lb)'].replace(',', ''))
                    max_weight = float(row['Max Weight (lb)'].replace(',', ''))
                    if weight > max_weight:
                        excess_amount = weight - max_weight
                        st.error(f"- {row['Component']}: Current {row['Weight (lb)']} lbs exceeds max {row['Max Weight (lb)']} lbs by {excess_amount:,.0f} lbs")
                st.stop()
    else:
        # For single configuration mode
        config = AIRCRAFT_CONFIG.get((aircraft_model, wing_type))
        if not config:
            st.error(f"No configuration found for {aircraft_model} with {wing_type} modification.")
            st.stop()
        
        try:
            s, b, e, h, sweep_25c, sfc, engines_orig, thrust_mult, ceiling, CL0, CLA, cdo, dcdo_flap1, dcdo_flap2, \
                dcdo_flap3, dcdo_gear, mu_to, mu_lnd, bow, mzfw, mrw, mtow, max_fuel, \
                taxi_fuel_default, reserve_fuel_default, mmo, VMO, clmax, clmax_1, clmax_2, m_climb, \
                v_climb, roc_min, m_descent, v_descent = config
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

# Update the last weight option
st.session_state.last_weight_option = weight_option