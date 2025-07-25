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

# Aircraft model selection
aircraft_types = ["CJ", "CJ1", "CJ1+", "M2", "CJ2", "CJ2+", "CJ3", "CJ3+"]
aircraft_model = st.selectbox("Aircraft Model", aircraft_types, index=aircraft_types.index("CJ1") if "CJ1" in aircraft_types else 0, key="aircraft_model")

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

# Display aircraft information
st.markdown("---")
with st.expander("Aircraft Specifications"):
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Basic Operating Weight", f"{int(bow):,} lb")
        st.metric("Maximum Zero Fuel Weight", f"{int(mzfw):,} lb")
        st.metric("Maximum Takeoff Weight", f"{int(mtow):,} lb")
    with col2:
        st.metric("Maximum Range Weight", f"{int(mrw):,} lb")
        st.metric("Maximum Fuel Capacity", f"{int(max_fuel):,} lb")
        st.metric("Maximum Payload", f"{int(max_payload):,} lb")

# Airport selection
st.markdown("---")
st.subheader('Flight Plan')

col1, col2 = st.columns(2)
with col1:
    departure_airport = st.selectbox(
        "Departure Airport",
        airport_display_names,
        index=next((i for i, name in enumerate(airport_display_names) if "KSZT" in name), 0)
    )
with col2:
    arrival_airport = st.selectbox(
        "Arrival Airport",
        airport_display_names,
        index=next((i for i, name in enumerate(airport_display_names) if "KSAN" in name), 0)
    )

# Get airport codes from display names
with st.expander("Airport Codes"):
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
        
        st.write(f"Departure Airport Code: {dep_airport_code}")
        st.write(f"Arrival Airport Code: {arr_airport_code}")
    except Exception as e:
        st.error(f"Error retrieving airport codes: {str(e)}")
        st.stop()

# Calculate and display range and bearing
try:
    # Get airport info using airport codes
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

    # Display range and bearing
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Distance", f"{distance_nm:.1f} NM")
    with col2:
        st.metric("Bearing", f"{bearing_deg:.1f}°")

except Exception as e:
    st.error(f"Error calculating route: {str(e)}")
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
    rw = bow + initial_payload + initial_fuel
    tow = rw - taxi_fuel_default
    if tow > mtow:
        initial_payload = mtow - (bow + initial_fuel) + taxi_fuel_default
        if initial_payload < 0:
            initial_payload = 0
            initial_fuel = mtow - bow + taxi_fuel_default
else:
    initial_payload = 0.0
    initial_fuel = 3440.0

# Prevent negative payloads
initial_payload = max(0, initial_payload)
initial_fuel = max(0, initial_fuel)

# Allow user adjustments
st.markdown("---")
st.subheader('Adjust Weights')
col1, col2 = st.columns(2)
with col1:
    payload_input = st.number_input(
        "Payload (lb)",
        min_value=0,
        max_value=int(max_payload),
        value=int(initial_payload),
        step=100,
        help=f"Maximum payload: {int(max_payload):,} lb",
        key="payload_input"
    )
with col2:
    fuel_input = st.number_input(
        "Initial Fuel (lb)",
        min_value=0,
        max_value=int(max_fuel),
        value=int(initial_fuel),
        step=100,
        help=f"Maximum fuel: {int(max_fuel):,} lb",
        key="fuel_input"
    )

# Calculate current weights using user inputs
zfw = bow + payload_input  # Zero Fuel Weight
rw = zfw + fuel_input  # Ramp Weight
tow = rw - taxi_fuel_default  # Takeoff Weight (after taxiing)
mission_fuel = fuel_input - reserve_fuel_default - taxi_fuel_default

# Add weight status table
st.markdown("---")
st.subheader('Weight Status')

# Create weight status dataframe
weight_df = pd.DataFrame({
    'Component': ['BOW', 'Payload', 'Initial Fuel', 'Reserve Fuel', 'Taxi Fuel', 'Mission Fuel', 'ZFW', 'Ramp Weight', 'Takeoff Weight'],
    'Weight (lb)': [
        f"{bow:,.0f}",
        f"{payload_input:,.0f}",
        f"{fuel_input:,.0f}",
        f"{reserve_fuel_default:,.0f}",
        f"{taxi_fuel_default:,.0f}",
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

# Add visual indicators for weights exceeding limits
weight_df['Status'] = weight_df.apply(
    lambda row: 
        f' Exceeds by {float(row["Weight (lb)"].replace(",", "")) - float(row["Max Weight (lb)"].replace(",", "")):,.0f} lbs. Adjust Payload, Fuel, or both' 
        if float(row['Weight (lb)'].replace(',', '')) > float(row['Max Weight (lb)'].replace(',', '')) 
        else ' OK',
    axis=1
)

# Reorder columns
weight_df = weight_df[['Component', 'Weight (lb)', 'Max Weight (lb)', 'Status']]

# Style the table to highlight exceeded limits
styled_df = weight_df.style.apply(
    lambda x: ['background-color: #ffcccc' if float(x['Weight (lb)'].replace(',', '')) > float(x['Max Weight (lb)'].replace(',', '')) else '' for i in x],
    axis=1
)

# Display the weight status table
st.table(styled_df)

# Check for any exceeded weight limits
exceeded_limits = weight_df['Status'].str.contains('Exceeds').any()
if exceeded_limits:
    error_message = "Weight limits exceeded! Please adjust the following:\n\n"
    for _, row in weight_df[weight_df['Status'].str.contains('Exceeds')].iterrows():
        excess_amount = float(row['Weight (lb)'].replace(',', '')) - float(row['Max Weight (lb)'].replace(',', ''))
        error_message += f"- {row['Component']}: Current {float(row['Weight (lb)'].replace(',', '')):,.0f} lbs exceeds max {float(row['Max Weight (lb)'].replace(',', '')):,.0f} lbs by {excess_amount:,.0f} lbs\n"
    st.error(error_message)
    st.stop()

# Other settings
st.markdown("---")
st.subheader('Flight Settings')

col1, col2, col3 = st.columns(3)
with col1:
    reserve_fuel = st.number_input(
        "Reserve Fuel (lb)",
        min_value=0,
        value=int(reserve_fuel_default),
        step=10,
        key="reserve_fuel_input"
    )
with col2:
    taxi_fuel = st.number_input(
        "Taxi Fuel (lb)",
        min_value=0,
        value=int(taxi_fuel_default),
        step=10
    )
with col3:
    cruise_altitude = st.number_input(
        "Cruise Altitude (ft)",
        min_value=0,
        max_value=int(ceiling),
        value=int(ceiling),
        step=1000,
        key="cruise_altitude"
    )

# Wing type selection
st.markdown("---")
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

# Sidebar for aircraft images
with st.sidebar:
    st.header("Aircraft Models")
    image_path_flatwing = f"images/flatwing_{aircraft_model}.jpg"
    image_path_tamarack = f"images/tamarack_{aircraft_model}.jpg"
    try:
        st.image(image_path_flatwing, caption=f"Flatwing {aircraft_model}", use_container_width=True)
    except FileNotFoundError:
        st.warning(f"Image not found: {image_path_flatwing}")
    try:
        st.image(image_path_tamarack, caption=f"Tamarack {aircraft_model}", use_container_width=True)
    except FileNotFoundError:
        st.warning(f"Image not found: {image_path_tamarack}")

if st.button("Run Simulation"):
    # Use user-adjusted values for simulation
    payload = payload_input
    fuel = fuel_input

    # Get airport info again in case it changed
    dep_info = airports_df[airports_df["ident"] == dep_airport_code]
    arr_info = airports_df[airports_df["ident"] == arr_airport_code]
    if dep_info.empty or arr_info.empty:
        st.error("Invalid airport code(s). Please check selection.")
        st.stop()

    dep_lat, dep_lon, elev_dep = dep_info.iloc[0][["latitude_deg", "longitude_deg", "elevation_ft"]]
    arr_lat, arr_lon, elev_arr = arr_info.iloc[0][["latitude_deg", "longitude_deg", "elevation_ft"]]

    config = AIRCRAFT_CONFIG.get((aircraft_model, def_config))
    if not config:
        st.error("Invalid aircraft config!")
        st.stop()

    tamarack_data = pd.DataFrame()
    tamarack_results = {}
    flatwing_data = pd.DataFrame()
    flatwing_results = {}

    if wing_type == "Comparison between Flatwing and Tamarack":
        if "Tamarack" in mods_available:
            tamarack_data, tamarack_results, dep_lat, dep_lon, arr_lat, arr_lon = run_simulation(
                dep_airport_code, arr_airport_code, aircraft_model, "Tamarack", takeoff_flap,
                payload, fuel, taxi_fuel, reserve_fuel, cruise_altitude,
                winds_temps_source, v1_cut_enabled)
        if "Flatwing" in mods_available:
            flatwing_data, flatwing_results, dep_lat, dep_lon, arr_lat, arr_lon = run_simulation(
                dep_airport_code, arr_airport_code, aircraft_model, "Flatwing", takeoff_flap,
                payload, fuel, taxi_fuel, reserve_fuel, cruise_altitude,
                winds_temps_source, v1_cut_enabled)
    elif wing_type == "Tamarack":
        tamarack_data, tamarack_results, dep_lat, dep_lon, arr_lat, arr_lon = run_simulation(
            dep_airport_code, arr_airport_code, aircraft_model, "Tamarack", takeoff_flap,
            payload, fuel, taxi_fuel, reserve_fuel, cruise_altitude,
            winds_temps_source, v1_cut_enabled)
    elif wing_type == "Flatwing":
        flatwing_data, flatwing_results, dep_lat, dep_lon, arr_lat, arr_lon = run_simulation(
            dep_airport_code, arr_airport_code, aircraft_model, "Flatwing", takeoff_flap,
            payload, fuel, taxi_fuel, reserve_fuel, cruise_altitude,
            winds_temps_source, v1_cut_enabled)

    if v1_cut_enabled:
        if not tamarack_data.empty:
            end_idx = tamarack_data[tamarack_data['Segment'] == 3].index[-1] if not tamarack_data[tamarack_data['Segment'] == 3].empty else 0
            tamarack_data = tamarack_data.iloc[:end_idx + 1]
        if not flatwing_data.empty:
            end_idx = flatwing_data[flatwing_data['Segment'] == 3].index[-1] if not flatwing_data[flatwing_data['Segment'] == 3].empty else 0
            flatwing_data = flatwing_data.iloc[:end_idx + 1]

    # Display the map first
    display_simulation_results(
        tamarack_data, tamarack_results,
        flatwing_data, flatwing_results,
        v1_cut_enabled,
        dep_lat, dep_lon, arr_lat, arr_lon,
        distance_nm, bearing_deg,
        winds_temps_source,
        cruise_altitude,
        dep_airport_code, arr_airport_code,
        fuel
    )

    # Ensure V-speeds are not printed prematurely by run_simulation
    # (This is handled by moving V-speed display logic to display_simulation_results if needed)