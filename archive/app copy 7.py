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

# Sidebar inputs
with st.sidebar:
    st.header("Flight Parameters")
    
    # Aircraft model selection
    aircraft_types = ["CJ", "CJ1", "CJ1+", "M2", "CJ2", "CJ2+", "CJ3", "CJ3+"]
    aircraft_model = st.selectbox("Aircraft Model", aircraft_types, 
                                 index=aircraft_types.index("CJ1") if "CJ1" in aircraft_types else 0,
                                 key="aircraft_model")

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

    # Get default configuration
    available_configs = [(a, m) for (a, m) in AIRCRAFT_CONFIG if a == aircraft_model]
    def_config = "Tamarack" if "Tamarack" in mods_available else "Flatwing"
    config = AIRCRAFT_CONFIG.get((aircraft_model, def_config))
    if not config:
        st.error(f"No configuration found for {aircraft_model} with {def_config} modification.")
        st.stop()

    # Extract configuration values
    try:
        s, b, e, h, sweep_25c, sfc, engines_orig, thrust_mult, ceiling, CL0, CLA, cdo, dcdo_flap1, dcdo_flap2, \
            dcdo_flap3, dcdo_gear, mu_to, mu_lnd, bow, mzfw, mrw, mtow, max_fuel, \
            taxi_fuel_default, reserve_fuel_default, mmo, VMO, clmax, clmax_1, clmax_2, m_climb, \
            v_climb, roc_min, m_descent, v_descent = config
        max_payload = mzfw - bow
    except ValueError as e:
        st.error(f"Error extracting configuration values: {str(e)}")
        st.stop()

    # Airport selection
    st.subheader('Flight Plan')
    departure_airport = st.selectbox(
        "Departure Airport",
        airport_display_names,
        index=next((i for i, name in enumerate(airport_display_names) if "KSZT" in name), 0)
    )
    arrival_airport = st.selectbox(
        "Arrival Airport",
        airport_display_names,
        index=next((i for i, name in enumerate(airport_display_names) if "KSAN" in name), 0)
    )

    # Weight mode selection
    weight_option = st.radio("Weight Configuration", [
        "Manual Input",
        "Max Fuel (Fill Tanks, Adjust Payload to MRW)",
        "Max Payload (Fill Payload to MZFW, Adjust Fuel to MRW)"
    ], index=0, key="weight_option")

    # Get initial values based on weight option
    if weight_option == "Max Fuel (Fill Tanks, Adjust Payload to MRW)":
        initial_fuel = float(max_fuel)
        initial_payload = float(min(max_payload, mrw - (bow + max_fuel)))
        rw = bow + initial_payload + initial_fuel
        tow = rw - taxi_fuel_default
        if tow > mtow:
            initial_fuel = mtow - (bow + initial_payload) + taxi_fuel_default
            if initial_fuel < 0:
                initial_fuel = 0
                initial_payload = mtow - bow + taxi_fuel_default
    elif weight_option == "Max Payload (Fill Payload to MZFW, Adjust Fuel to MRW)":
        initial_payload = float(max_payload)
        initial_fuel = float(min(max_fuel, mrw - (bow + max_payload)))
        if initial_fuel == max_fuel:
            initial_payload = float(min(max_payload, mrw - (bow + max_fuel)))
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
    col1, col2 = st.columns(2)
    with col1:
        payload_input_f = st.number_input(
            "Flatwing Payload (lb)",
            min_value=0,
            max_value=int(max_payload),
            value=st.session_state.initial_values.get('payload', int(initial_payload)),
            step=100,
            help=f"Maximum payload: {int(max_payload):,} lb",
            key="payload_input_flatwing"
        )
        reserve_fuel_f = st.number_input(
            "Flatwing Reserve Fuel (lb)",
            min_value=0,
            value=st.session_state.initial_values.get('reserve_fuel', int(reserve_fuel_default)),
            step=10,
            key="reserve_fuel_input_flatwing"
        )
        cruise_altitude_f = st.number_input(
            "Flatwing Cruise Altitude (ft)",
            min_value=0,
            max_value=int(ceiling),
            value=st.session_state.initial_values.get('cruise_altitude', int(ceiling)),
            step=1000,
            key="cruise_altitude_input_flatwing"
        )
    with col2:
        fuel_input_f = st.number_input(
            "Flatwing Fuel (lb)",
            min_value=0,
            max_value=int(max_fuel),
            value=st.session_state.initial_values.get('fuel', int(initial_fuel)),
            step=100,
            help=f"Maximum fuel: {int(max_fuel):,} lb",
            key="fuel_input_flatwing"
        )
        taxi_fuel_f = st.number_input(
            "Flatwing Taxi Fuel (lb)",
            min_value=0,
            value=st.session_state.initial_values.get('taxi_fuel', int(taxi_fuel_default)),
            step=10,
            key="taxi_fuel_input_flatwing"
        )

    # Weight inputs - Tamarack
    st.subheader('Tamarack Weight Adjustment')
    col1, col2 = st.columns(2)
    with col1:
        payload_input_t = st.number_input(
            "Tamarack Payload (lb)",
            min_value=0,
            max_value=int(max_payload),
            value=st.session_state.initial_values.get('payload', int(initial_payload)),
            step=100,
            help=f"Maximum payload: {int(max_payload):,} lb",
            key="payload_input_tamarack"
        )
        reserve_fuel_t = st.number_input(
            "Tamarack Reserve Fuel (lb)",
            min_value=0,
            value=st.session_state.initial_values.get('reserve_fuel', int(reserve_fuel_default)),
            step=10,
            key="reserve_fuel_input_tamarack"
        )
        cruise_altitude_t = st.number_input(
            "Tamarack Cruise Altitude (ft)",
            min_value=0,
            max_value=int(ceiling),
            value=st.session_state.initial_values.get('cruise_altitude', int(ceiling)),
            step=1000,
            key="cruise_altitude_input_tamarack"
        )
    with col2:
        fuel_input_t = st.number_input(
            "Tamarack Fuel (lb)",
            min_value=0,
            max_value=int(max_fuel),
            value=st.session_state.initial_values.get('fuel', int(initial_fuel)),
            step=100,
            help=f"Maximum fuel: {int(max_fuel):,} lb",
            key="fuel_input_tamarack"
        )
        taxi_fuel_t = st.number_input(
            "Tamarack Taxi Fuel (lb)",
            min_value=0,
            value=st.session_state.initial_values.get('taxi_fuel', int(taxi_fuel_default)),
            step=10,
            key="taxi_fuel_input_tamarack"
        )

    # Wing type selection
    st.subheader('Aircraft Configuration')
    wing_type = st.radio("Wing Type", ["Flatwing", "Tamarack", "Comparison between Flatwing and Tamarack"], index=0, key="wing_type")
    if wing_type != "Comparison between Flatwing and Tamarack" and wing_type not in mods_available:
        st.error(f"Wing type '{wing_type}' is not available for aircraft model {aircraft_model}. Available options: {mods_available}")
        st.stop()

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
    if wing_type == "Comparison between Flatwing and Tamarack":
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
    if wing_type == "Comparison between Flatwing and Tamarack":
        # Flatwing weights
        zfw_f = bow + payload_f  # Zero Fuel Weight
        rw_f = zfw_f + fuel_f  # Ramp Weight
        tow_f = rw_f - taxi_fuel_f  # Takeoff Weight (after taxiing)
        mission_fuel_f = fuel_f - reserve_fuel_f - taxi_fuel_f
        
        # Tamarack weights
        zfw_t = bow + payload_t  # Zero Fuel Weight
        rw_t = zfw_t + fuel_t  # Ramp Weight
        tow_t = rw_t - taxi_fuel_t  # Takeoff Weight (after taxiing)
        mission_fuel_t = fuel_t - reserve_fuel_t - taxi_fuel_t
    elif wing_type == "Flatwing":
        zfw_f = bow + payload_f  # Zero Fuel Weight
        rw_f = zfw_f + fuel_f  # Ramp Weight
        tow_f = rw_f - taxi_fuel_f  # Takeoff Weight (after taxiing)
        mission_fuel_f = fuel_f - reserve_fuel_f - taxi_fuel_f
        
        zfw_t = 0
        rw_t = 0
        tow_t = 0
        mission_fuel_t = 0
    elif wing_type == "Tamarack":
        zfw_f = 0
        rw_f = 0
        tow_f = 0
        mission_fuel_f = 0
        
        zfw_t = bow + payload_t  # Zero Fuel Weight
        rw_t = zfw_t + fuel_t  # Ramp Weight
        tow_t = rw_t - taxi_fuel_t  # Takeoff Weight (after taxiing)
        mission_fuel_t = fuel_t - reserve_fuel_t - taxi_fuel_t

    # Get configurations based on wing type
    if wing_type == "Comparison between Flatwing and Tamarack":
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
                f"{payload_f:,.0f}",
                f"{fuel_f:,.0f}",
                f"{reserve_fuel_f:,.0f}",
                f"{taxi_fuel_f:,.0f}",
                f"{mission_fuel_f:,.0f}",
                f"{zfw_f:,.0f}",
                f"{rw_f:,.0f}",
                f"{tow_f:,.0f}"
            ],
            'Max Weight (lb)': [
                f"{bow_f:,.0f}",
                f"{max_payload_f:,.0f}",
                f"{max_fuel_f:,.0f}",
                f"{reserve_fuel_default_f:,.0f}",
                f"{taxi_fuel_default_f:,.0f}",
                f"{mission_fuel_f:,.0f}",
                f"{mzfw_f:,.0f}",
                f"{mrw_f:,.0f}",
                f"{mtow_f:,.0f}"
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
                f"{payload_t:,.0f}",
                f"{fuel_t:,.0f}",
                f"{reserve_fuel_t:,.0f}",
                f"{taxi_fuel_t:,.0f}",
                f"{mission_fuel_t:,.0f}",
                f"{zfw_t:,.0f}",
                f"{rw_t:,.0f}",
                f"{tow_t:,.0f}"
            ],
            'Max Weight (lb)': [
                f"{bow_t:,.0f}",
                f"{max_payload_t:,.0f}",
                f"{max_fuel_t:,.0f}",
                f"{reserve_fuel_default_t:,.0f}",
                f"{taxi_fuel_default_t:,.0f}",
                f"{mission_fuel_t:,.0f}",
                f"{mzfw_t:,.0f}",
                f"{mrw_t:,.0f}",
                f"{mtow_t:,.0f}"
            ]
        })

        # Display weight status tables before simulation
        st.markdown("---")
        st.subheader('Weight Status')
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Flatwing Weight Status")
            styled_df = weight_df.style.apply(
                lambda x: ['background-color: #ffcccc' if float(x['Weight (lb)'].replace(',', '')) > float(x['Max Weight (lb)'].replace(',', '')) else '' for i in x],
                axis=1
            )
            st.table(styled_df)
            
            # Check for any exceeded weight limits
            weight_exceeded = weight_df.apply(
                lambda row: float(row['Weight (lb)'].replace(',', '')) > float(row['Max Weight (lb)'].replace(',', '')),
                axis=1
            ).any()
            
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
            styled_df_tamarack = weight_df_tamarack.style.apply(
                lambda x: ['background-color: #ffcccc' if float(x['Weight (lb)'].replace(',', '')) > float(x['Max Weight (lb)'].replace(',', '')) else '' for i in x],
                axis=1
            )
            st.table(styled_df_tamarack)
            
            # Check for any exceeded weight limits
            weight_exceeded_t = weight_df_tamarack.apply(
                lambda row: float(row['Weight (lb)'].replace(',', '')) > float(row['Max Weight (lb)'].replace(',', '')),
                axis=1
            ).any()
            
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
            payload = payload_f
            fuel = fuel_f
            taxi_fuel = taxi_fuel_f
            reserve_fuel = reserve_fuel_f
            mission_fuel = mission_fuel_f
            zfw = zfw_f
            rw = rw_f
            tow = tow_f
        else:  # Tamarack
            payload = payload_t
            fuel = fuel_t
            taxi_fuel = taxi_fuel_t
            reserve_fuel = reserve_fuel_t
            mission_fuel = mission_fuel_t
            zfw = zfw_t
            rw = rw_t
            tow = tow_t

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
                f"{bow:,.0f}",
                f"{max_payload:,.0f}",
                f"{max_fuel:,.0f}",
                f"{reserve_fuel_default:,.0f}",
                f"{taxi_fuel_default:,.0f}",
                f"{mission_fuel:,.0f}",
                f"{mzfw:,.0f}",
                f"{mrw:,.0f}",
                f"{mtow:,.0f}"
            ]
        })

        # Display weight status table before simulation
        st.markdown("---")
        st.subheader('Weight Status')
        styled_df = weight_df.style.apply(
            lambda x: ['background-color: #ffcccc' if float(x['Weight (lb)'].replace(',', '')) > float(x['Max Weight (lb)'].replace(',', '')) else '' for i in x],
            axis=1
        )
        st.table(styled_df)
        
        # Check for any exceeded weight limits
        weight_exceeded = weight_df.apply(
            lambda row: float(row['Weight (lb)'].replace(',', '')) > float(row['Max Weight (lb)'].replace(',', '')),
            axis=1
        ).any()
        
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

    if wing_type == "Comparison between Flatwing and Tamarack":
        if "Tamarack" in mods_available:
            tamarack_data, tamarack_results, dep_lat, dep_lon, arr_lat, arr_lon = run_simulation(
                dep_airport_code, arr_airport_code, aircraft_model, "Tamarack", takeoff_flap,
                payload_t, fuel_t, taxi_fuel_t, reserve_fuel_t, cruise_altitude_t,
                winds_temps_source, v1_cut_enabled)
        if "Flatwing" in mods_available:
            flatwing_data, flatwing_results, dep_lat, dep_lon, arr_lat, arr_lon = run_simulation(
                dep_airport_code, arr_airport_code, aircraft_model, "Flatwing", takeoff_flap,
                payload_f, fuel_f, taxi_fuel_f, reserve_fuel_f, cruise_altitude_f,
                winds_temps_source, v1_cut_enabled)
    elif wing_type == "Tamarack":
        tamarack_data, tamarack_results, dep_lat, dep_lon, arr_lat, arr_lon = run_simulation(
            dep_airport_code, arr_airport_code, aircraft_model, "Tamarack", takeoff_flap,
            payload_t, fuel_t, taxi_fuel_t, reserve_fuel_t, cruise_altitude_t,
            winds_temps_source, v1_cut_enabled)
    elif wing_type == "Flatwing":
        flatwing_data, flatwing_results, dep_lat, dep_lon, arr_lat, arr_lon = run_simulation(
            dep_airport_code, arr_airport_code, aircraft_model, "Flatwing", takeoff_flap,
            payload_f, fuel_f, taxi_fuel_f, reserve_fuel_f, cruise_altitude_f,
            winds_temps_source, v1_cut_enabled)

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