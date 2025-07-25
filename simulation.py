from math import radians, sin, cos, sqrt, degrees, pi, tan

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from aircraft_config import AIRCRAFT_CONFIG
from flight_physics import atmos, vspeeds, physics, predict_roc, next_step_altitude
from utils import load_airports

def compute_segment_fuel_remaining(total_initial_fuel, fuel_burn_sequence):
    """
    Compute remaining fuel after each mission segment.

    Args:
        total_initial_fuel (float): Initial fuel at mission start.
        fuel_burn_sequence (dict): Dict of segment names to fuel burned (e.g., "Climb Fuel (lb)": 500)

    Returns:
        dict: Segment keys mapped to computed fuel remaining (e.g., "Fuel Remaining After Climb (lb)": 7200)
    """
    remaining_fuel = total_initial_fuel
    fuel_remaining_dict = {}

    segment_map = [
        ("Takeoff", None),
        ("Climb", "Climb Fuel (lb)"),
        ("Cruise", "Cruise Fuel (lb)"),
        ("Descent", "Descent Fuel (lb)")
    ]

    for segment, fuel_key in segment_map:
        if fuel_key and fuel_key in fuel_burn_sequence:
            remaining_fuel -= fuel_burn_sequence[fuel_key]

    fuel_remaining_dict["Fuel Remaining After Descent (lb)"] = round(remaining_fuel)
    fuel_remaining_dict["Fuel Remaining After Landing (lb)"] = round(remaining_fuel)
    fuel_remaining_dict["Fuel Remaining Final (lb)"] = round(remaining_fuel)

    return fuel_remaining_dict

# --- Helper Functions ---
def haversine_with_bearing(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> tuple[float, float]:
    """
    Calculate the distance and bearing between two points on Earth using the haversine formula.

    Args:
        lat1: Latitude of the first point in degrees.
        lon1: Longitude of the first point in degrees.
        lat2: Latitude of the second point in degrees.
        lon2: Longitude of the second point in degrees.

    Returns:
        tuple: A tuple containing the distance in nautical miles and the bearing in degrees.
    """
    earth_radius_nm = 3437.75
    lat1_rad = radians(lat1)
    lat2_rad = radians(lat2)
    lon1_rad = radians(lon1)
    lon2_rad = radians(lon2)
    delta_lat = lat2_rad - lat1_rad
    delta_lon = lon2_rad - lon1_rad
    a = sin(delta_lat / 2) ** 2 + cos(lat1_rad) * cos(lat2_rad) * sin(delta_lon / 2) ** 2
    c = 2 * np.arcsin(sqrt(a))
    distance = earth_radius_nm * c
    y = sin(delta_lon) * cos(lat2_rad)
    x = cos(lat1_rad) * sin(lat2_rad) - sin(lat1_rad) * cos(lat2_rad) * cos(delta_lon)
    if x == 0 and y == 0:
        initial_bearing_rad = 0
    else:
        initial_bearing_rad = np.arctan2(y, x)
        if initial_bearing_rad < 0:
            initial_bearing_rad += 2 * pi
    bearing = (degrees(initial_bearing_rad) + 360) % 360
    return distance, bearing

def get_intermediate_points(dep_lat, dep_lon, arr_lat, arr_lon, num_points=5):
    """
    Calculate intermediate points along the great circle route between two airports.

    Args:
        dep_lat: Departure latitude in degrees.
        dep_lon: Departure longitude in degrees.
        arr_lat: Arrival latitude in degrees.
        arr_lon: Arrival longitude in degrees.
        num_points: Number of intermediate points to calculate.

    Returns:
        list: List of (latitude, longitude) tuples along the route.
    """
    points = [(dep_lat, dep_lon)]
    total_dist, bearing = haversine_with_bearing(dep_lat, dep_lon, arr_lat, arr_lon)
    step_dist = total_dist / (num_points + 1)

    earth_radius_nm = 3437.75
    current_lat = dep_lat
    current_lon = dep_lon

    for i in range(num_points):
        fraction = (i + 1) * step_dist / total_dist
        lat1_rad = radians(current_lat)
        lon1_rad = radians(current_lon)
        bearing_rad = radians(bearing)

        a = sin((1 - fraction) * total_dist / earth_radius_nm) / sin(total_dist / earth_radius_nm)
        b = sin(fraction * total_dist / earth_radius_nm) / sin(total_dist / earth_radius_nm)
        x = a * cos(lat1_rad) * cos(lon1_rad) + b * cos(radians(arr_lat)) * cos(radians(arr_lon))
        y = a * cos(lat1_rad) * sin(lon1_rad) + b * cos(radians(arr_lat)) * sin(radians(arr_lon))
        z = a * sin(lat1_rad) + b * sin(radians(arr_lat))

        lat_rad = np.arctan2(z, sqrt(x**2 + y**2))
        lon_rad = np.arctan2(y, x)

        new_lat = degrees(lat_rad)
        new_lon = degrees(lon_rad)

        points.append((new_lat, new_lon))
        current_lat, current_lon = new_lat, new_lon

    points.append((arr_lat, arr_lon))
    return points

def interpolate_winds_temps(altitude, winds_temps_data):
    """
    Interpolate wind speed, direction, and temperature at a given altitude.

    Args:
        altitude: Altitude in feet.
        winds_temps_data: Dictionary of flight level data (wind_dir, wind_speed, temp).

    Returns:
        tuple: (wind_direction, wind_speed, temperature) interpolated for the altitude.
    """
    flight_level = int(altitude / 100)
    altitudes = sorted(winds_temps_data.keys())

    if altitude <= altitudes[0]:
        wind_dir, wind_speed, temp = winds_temps_data[altitudes[0]]
        return wind_dir, wind_speed, temp
    if altitude >= altitudes[-1]:
        wind_dir, wind_speed, temp = winds_temps_data[altitudes[-1]]
        return wind_dir, wind_speed, temp

    lower_alt = max([a for a in altitudes if a <= altitude])
    upper_alt = min([a for a in altitudes if a >= altitude])

    if lower_alt == upper_alt:
        wind_dir, wind_speed, temp = winds_temps_data[lower_alt]
        return wind_dir, wind_speed, temp

    fraction = (altitude - lower_alt) / (upper_alt - lower_alt)
    lower_data = winds_temps_data[lower_alt]
    upper_data = winds_temps_data[upper_alt]

    wind_dir = lower_data[0] + fraction * (upper_data[0] - lower_data[0])
    wind_speed = lower_data[1] + fraction * (upper_data[1] - lower_data[1])
    temp = lower_data[2] + fraction * (upper_data[2] - lower_data[2])

    return wind_dir, wind_speed, temp

# --- Simulation Logic ---
def run_simulation(
    dep_airport: str,
    arr_airport: str,
    aircraft: str,
    mod: str,
    takeoff_flap_setting: int,
    payload: float,
    initial_fuel: float,
    taxi_fuel: float,
    reserve_fuel: float,
    cruise_alt: int,
    winds_temps_source: str,
    v1_cut_enabled: bool,
):
    """Simulate a flight between two airports.
    
    Args:
        dep_airport: Departure airport ICAO code.
        arr_airport: Arrival airport ICAO code.
        aircraft: Aircraft model.
        mod: Aircraft modification (e.g., "Flatwing" or "Tamarack").
        takeoff_flap_setting: Takeoff flap setting (0 for 0°, 1 for 15°).
        payload: Payload weight in pounds.
        initial_fuel: Initial fuel load in pounds.
        taxi_fuel: Fuel used for taxiing in pounds.
        reserve_fuel: Reserve fuel in pounds.
        cruise_alt: Cruise altitude in feet.
        winds_temps_source: Source for winds and temperatures.
        v1_cut_enabled: Whether V1 cut is enabled.

    Returns:
        tuple: (flight_data, results, dep_lat, dep_lon, arr_lat, arr_lon)
    """
    # Initialize variables
    alt = 0
    d_alt = 0
    vkias = 0
    vktas = 0
    v_true_fps = 0
    w = 0
    thrust = 0
    drag = 0
    roc_fpm = 0
    m = 0
    dist_ft = 0
    gamma = 0
    mission_fuel_remain = 0
    sigma = 0
    delta = 0
    c = 0
    s = 0
    dep_latitude = 0
    dep_longitude = 0
    arr_latitude = 0
    arr_longitude = 0
    clmax = 0
    clmax_1 = 0
    clmax_2 = 0

    # Variables to track takeoff distances and climb data
    takeoff_roll_dist = 0  # Segment 0: Distance to VR
    dist_to_35ft = 0  # Segment 1: Distance from VR to 35 ft
    dist_to_400ft = 0  # Segment 2: Distance from 35 ft to 400 ft
    dist_to_1500ft = 0  # Segment 3: Distance from 400 ft to 1500 ft
    segment1_gradient = 0  # Gradient for segment 1
    segment2_gradient = 0  # Gradient for segment 2
    segment3_gradient = 0  # Gradient for segment 3
    climb_time = 0
    climb_dist = 0
    climb_fuel = 0
    first_level_off_alt = None  # Track the first level-off altitude

    # Variables for tracking step altitudes in segment 6
    step_altitudes = []  # List to store step altitudes
    last_segment_6_step = -10  # Track the last step when segment 6 was entered
    debounce_steps = 10  # Minimum steps between segment 6 entries to avoid glitches

    time_data = []
    alt_data = []
    dist_data = []
    vktas_data = []
    vkias_data = []
    roc_data = []
    thrust_data = []
    drag_data = []
    segment_data = []
    mach_data = []
    gradient_data = []  # To store gradient values

    alt_tolerance = 100

    ac = AIRCRAFT_CONFIG[(aircraft, mod)]
    s, b, e, h, _, sfc, engines_orig, thrust_mult, _, _, _, cdo, dcdo_flap1, dcdo_flap2, \
        dcdo_flap3, dcdo_gear, mu_to, mu_lnd, bow, mzfw, mrw, mtow, max_fuel, \
        taxi_fuel_default, reserve_fuel_default, mmo, _, clmax, clmax_1, clmax_2, m_climb, \
        v_climb, roc_min, m_descent, v_descent = ac

    wind = 0  # Will be updated based on winds aloft
    payload = payload
    fuel_start = initial_fuel
    flap = takeoff_flap_setting
    v1_cut = 1 if v1_cut_enabled else 0
    engines = engines_orig  # Store original engines value, may be modified by V1 cut
    reserve_fuel = reserve_fuel
    m_cruise = 0.7
    alt_goal = cruise_alt
    rod = -2000
    rod_u_10k = -1500
    rod_approach = -700
    
    airports = load_airports()
    try:
        dep_data = airports[airports['ident'] == dep_airport].iloc[0]
        arr_data = airports[airports['ident'] == arr_airport].iloc[0]
        dep_latitude = dep_data['latitude_deg']
        dep_longitude = dep_data['longitude_deg']
        dep_elev = dep_data['elevation_ft']
        arr_latitude = arr_data['latitude_deg']
        arr_longitude = arr_data['longitude_deg']
        arr_elev = dep_data['elevation_ft']  # Assume same elevation for simplicity, or fix to arr_data
    except IndexError:
        return pd.DataFrame(), {"error": "Invalid airport code(s)."}, 0, 0, 0, 0  # Return empty results if airport lookup fails

    total_dist, bearing = haversine_with_bearing(dep_latitude, dep_longitude, arr_latitude, arr_longitude)
    remaining_dist = total_dist
    alt_to = dep_elev
    alt_land = arr_elev
    alt = alt_to

    # Get intermediate points along the route for sampling winds and temps
    route_points = get_intermediate_points(dep_latitude, dep_longitude, arr_latitude, arr_longitude, num_points=5)
    point_distances = [0]
    cumulative_dist = 0
    for i in range(1, len(route_points)):
        dist, _ = haversine_with_bearing(route_points[i-1][0], route_points[i-1][1], route_points[i][0], route_points[i][1])
        cumulative_dist += dist
        point_distances.append(cumulative_dist)

    # Placeholder winds and temps aloft data (wind direction in degrees, speed in knots, temp in °C)
    winds_temps_data = {
        "Current Conditions": {
            18000: (310, 30, -20),  # FL180
            30000: (310, 40, -35),  # FL300
            39000: (310, 50, -45),  # FL390
        },
        "Summer Average": {
            18000: (270, 15, -10),
            30000: (270, 20, -25),
            39000: (270, 25, -30),
        },
        "Winter Average": {
            18000: (320, 35, -25),
            30000: (320, 45, -40),
            39000: (320, 55, -50),
        }
    }
    selected_winds_temps = winds_temps_data[winds_temps_source]

    v_u_10k = 200
    a = b ** 2 / s * (1 + 1.9 * h / b)
    k = 1 / (3.14159 * e * a)
    max_payload = mzfw - bow
    zfw = bow + payload  # Zero Fuel Weight
    rw = bow + payload + fuel_start  # Ramp Weight
    tow = rw - taxi_fuel  # Takeoff Weight (after taxiing)
    mission_fuel = fuel_start - reserve_fuel - taxi_fuel
    mission_fuel_remain = mission_fuel

    # Check constraints and collect all exceedance messages
    exceedances = []
    if zfw > mzfw:
        exceedances.append(f"{mod}: Zero Fuel Weight exceeds MZFW of {int(mzfw)} lb by {int(zfw - mzfw)} lb. Reduce payload.")
    if payload > max_payload:
        exceedances.append(f"{mod}: Payload exceeds maximum payload of {int(max_payload)} lb by {int(payload - max_payload)} lb.")
    if fuel_start > max_fuel:
        exceedances.append(f"{mod}: Fuel exceeds maximum fuel capacity of {int(max_fuel)} lb by {int(fuel_start - max_fuel)} lb.")
    if tow > mtow:
        exceedances.append(f"{mod}: Takeoff Weight exceeds MTOW of {int(mtow)} lb by {int(tow - mtow)} lb.")
    if rw > mrw:
        exceedances.append(f"{mod}: Ramp Weight exceeds MRW of {int(mrw)} lb by {int(rw - mrw)} lb.")
    if exceedances:
        return pd.DataFrame(), {"exceedances": exceedances}, dep_latitude, dep_longitude, arr_latitude, arr_longitude

    thrust_factor = 1
    v_true_fps = 0
    roc_fpm = 0
    w = tow
    dist_ft = 0
    gamma = 0
    m = 0
    vkias = 0
    segment = 0
    p = 0
    climb_trigger = 0
    land_trigger = 0
    to_flag = 0
    climb_fuel_flag = 0
    vspeed_flag = 0
    approach_vspeed_flag = 0
    descent_threshold = 0.0031 * (alt_goal - alt_land) - 9.7404
    next_step_alt = 0
    t = 0
    fuel_burned = 0
    climb_fuel = 0
    climb_time = 0
    climb_dist = 0
    leveloff = 0
    fuel_start_descent = 0
    t_start_descent = 0
    cruise_time = 0
    cruise_dist = 0
    cruise_fuel = 0
    max_m_reached = 0
    dist_at_35 = 0
    dist_touchdown = 0
    dist_land = 0
    descent_dist = 0
    descent_fuel = 0
    fob = 0
    last_segment = -1

    # Variables to track descent and landing data
    dist_land_35 = 0
    dist_land = 0
    dist_ground_roll = 0
    landing_start_time = 0
    landing_start_dist = 0
    descent_start_time = 0
    descent_start_fuel = 0
    descent_start_dist = 0
    descent_start_alt = 0

    # Variables to track segment weights
    takeoff_start_weight = w
    climb_start_weight = 0
    cruise_start_weight = 0
    descent_start_weight = 0
    landing_start_weight = 0
    takeoff_end_weight = 0
    climb_end_weight = 0
    cruise_end_weight = 0
    descent_end_weight = 0
    landing_end_weight = 0

    # Initialize final_results with default structure
    final_results = {
        "Takeoff Roll Dist (ft)": None,
        "Takeoff Start Weight (lb)": None,
        "Takeoff End Weight (lb)": None,
        "Dist to 35 ft (ft)": None,
        "Segment 1 Gradient (%)": None,
        "Dist to 400 ft (ft)": None,
        "Segment 2 Gradient (%)": None,
        "Dist to 1500 ft (ft)": None,
        "Segment 3 Gradient (%)": None,
        "Climb Time (min)": None,
        "Climb Dist (NM)": None,
        "Climb Start Weight (lb)": None,
        "Climb End Weight (lb)": None,
        "Climb Fuel (lb)": None,
        "Fuel Remaining After Takeoff (lb)": None,
        "Fuel Remaining After Climb (lb)": None,
        "Cruise Time (min)": None,
        "Cruise Dist (NM)": None,
        "Cruise Start Weight (lb)": None,
        "Cruise End Weight (lb)": None,
        "Cruise Fuel (lb)": None,
        "Cruise VKTAS (knots)": None,
        "Cruise - First Level-Off Alt (ft)": None,
        "Step Altitudes (ft)": None,
        "Descent Time (min)": None,
        "Descent Dist (NM)": None,
        "Descent Start Weight (lb)": None,
        "Descent End Weight (lb)": None,
        "Descent Fuel (lb)": None,
        "Fuel Remaining After Descent (lb)": None,
        "Landing - Dist from 35 ft to Stop (ft)": None,
        "Landing - Ground Roll (ft)": None,
        "Landing Start Weight (lb)": None,
        "Landing End Weight (lb)": None,
        "Fuel Remaining After Landing (lb)": None,
        "Total Time (min)": None,
        "Total Dist (NM)": None,
        "Total Fuel Burned (lb)": None,
        "Fuel Remaining (lb)": None,
        "Takeoff V-Speeds": None,
        "Approach V-Speeds": None,
        "V1 Cut": v1_cut_enabled,
        "First Level-Off Alt (ft)": None,
    }

    # Calculate V-speeds at the start
    try:
        # Takeoff V-speeds
        takeoff_weight = tow  # Use takeoff weight
        vr, v1, v2, v3, _, _ = vspeeds(
            w=takeoff_weight,
            s=s,
            clmax=clmax,
            clmax_1=clmax_1,
            clmax_2=clmax_2,
            delta=1.0,  # Sea level pressure ratio
            m=0.2,  # Initial Mach guess
            flap=takeoff_flap_setting,
            segment=0
        )
        takeoff_vspeeds = {
            "Weight": int(takeoff_weight),
            "VR": round(float(vr), 1) if vr is not None else None,
            "V1": round(float(v1), 1) if v1 is not None else None,
            "V2": round(float(v2), 1) if v2 is not None else None,
            "V3": round(float(v3), 1) if v3 is not None else None
        }
        final_results["Takeoff V-Speeds"] = takeoff_vspeeds

        # Approach V-speeds (using estimated landing weight)
        landing_weight = tow * 0.85  # Approximate landing weight
        _, _, _, _, vapp, vref = vspeeds(
            w=landing_weight,
            s=s,
            clmax=clmax,
            clmax_1=clmax_1,
            clmax_2=clmax_2,
            delta=1.0,
            m=0.2,
            flap=2,  # Landing flap setting
            segment=12
        )
        approach_vspeeds = {
            "Weight": int(landing_weight),
            "VAPP": round(float(vapp), 1) if vapp is not None else None,
            "VREF": round(float(vref), 1) if vref is not None else None
        }
        final_results["Approach V-Speeds"] = approach_vspeeds

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        final_results["Takeoff V-Speeds"] = {"error": f"Failed to calculate: {str(e)}"}
        final_results["Approach V-Speeds"] = {"error": f"Failed to calculate: {str(e)}"}

    fuel_burn_history = []

    while segment != 14:
        if mission_fuel_remain < 0 and alt > alt_land:
            final_results = {"error": f"Not Enough Fuel for {mod}."}
            return pd.DataFrame(), final_results, dep_latitude, dep_longitude, arr_latitude, arr_longitude

        # Reset vspeed_flag when entering segment 0, 12, or 13
        if segment in (0, 12, 13) and vspeed_flag != 0:
            vspeed_flag = 0
            
        if segment in (0, 12, 13) and vspeed_flag == 0:
            try:
                # Calculate V-speeds based on current segment
                if segment == 0:  # Takeoff
                    vr, v1, v2, v3, _, _ = vspeeds(w, s, clmax, clmax_1, clmax_2, delta, m, flap, segment)
                    
                    takeoff_vspeeds = {
                        "Weight": int(w),
                        "VR": round(float(vr), 1) if vr is not None else None,
                        "V1": round(float(v1), 1) if v1 is not None else None,
                        "V2": round(float(v2), 1) if v2 is not None else None,
                        "V3": round(float(v3), 1) if v3 is not None else None
                    }
                    final_results["Takeoff V-Speeds"] = takeoff_vspeeds
                
                elif segment == 12:  # Approach
                    # Set landing flap setting for approach
                    approach_flap = 2  # Assuming 2 is the landing flap setting
                    _, _, _, _, vapp, vref = vspeeds(w, s, clmax, clmax_1, clmax_2, delta, m, approach_flap, 12)
                    
                    approach_vspeeds = {
                        "Weight": int(w),
                        "VAPP": round(float(vapp), 1) if vapp is not None else None,
                        "VREF": round(float(vref), 1) if vref is not None else None
                    }
                    final_results["Approach V-Speeds"] = approach_vspeeds
                
                # Set flag to indicate V-speeds have been calculated for this segment
                vspeed_flag = 1
                
            except Exception as e:
                import traceback
                print(traceback.format_exc())
                
                # Store error information for debugging
                error_key = "Takeoff V-Speeds" if segment == 0 else "Approach V-Speeds"
                final_results[error_key] = {
                    "error": f"Failed to calculate V-speeds: {str(e)}",
                    "segment": segment,
                    "weight": w,
                    "wing_area": s,
                    "clmax": clmax,
                    "clmax_1": clmax_1,
                    "clmax_2": clmax_2,
                    "delta": delta,
                    "mach": m,
                    "flap": flap
                }

        # Set time increment based on segment
        if segment in (0, 13):
            t_inc = 0.1
        elif segment == 1:
            t_inc = 0.5
        elif segment == 12:
            t_inc = 0.5
        elif segment == 3:
            t_inc = 1
        elif segment == 6:
            t_inc = 5
        elif segment == 7:
            t_inc = 10
        elif segment == 8:
            t_inc = 5
        else:
            t_inc = 1

        # Set speed and ROC goals based on segment
        speed_goal = 0
        roc_goal = 0
        if segment == 0:
            speed_goal = vr
        elif segment == 1:
            speed_goal = v1
        elif segment == 2:
            speed_goal = v2
        elif segment == 3:
            speed_goal = v3
        elif segment == 4:
            speed_goal = v_climb
            roc_goal = roc_min
        elif segment == 5:
            speed_goal = m_climb
            roc_goal = roc_min
        elif segment in (6, 7):
            if alt <= 10100:
                speed_goal = v_u_10k
            else:
                speed_goal = m_cruise
        elif segment == 8:
            if alt <= 10100:
                speed_goal = v_u_10k
                roc_goal = rod_u_10k
            else:
                speed_goal = m_descent
                roc_goal = rod
        elif segment == 9:
            speed_goal = v_descent
            roc_goal = rod
        elif segment == 10:
            speed_goal = v_u_10k
            roc_goal = rod_u_10k
        elif segment == 11:
            speed_goal = vapp
            roc_goal = rod_approach
        elif segment == 12:
            speed_goal = vref
            roc_goal = rod_approach
        elif segment == 13:
            speed_goal = 0
            thrust_factor = 0.05

        # Find the closest route point to the current distance
        current_dist = dist_ft / 6076.12  # Convert to NM
        closest_point_idx = min(range(len(point_distances)), key=lambda i: abs(point_distances[i] - current_dist))
        point_lat, point_lon = route_points[closest_point_idx]

        # Special case for V1 cut: terminate after 100 NM
        if v1_cut == 1 and current_dist >= 100:
            segment = 14  # Force termination
            break

        # Interpolate winds and temperatures at the current altitude
        wind_dir, wind_speed, temp = interpolate_winds_temps(alt, selected_winds_temps)

        # Compute ISA temperature at current altitude for density altitude adjustment
        isa_temp = 15 - 0.0019812 * alt  # ISA temperature lapse rate (approx)
        isa_diff = temp - isa_temp  # Deviation from ISA

        # Compute wind component affecting ground speed
        heading_rad = radians(bearing)
        wind_dir_rad = radians(wind_dir)
        wind_component = wind_speed * cos(wind_dir_rad - heading_rad)  # Positive for tailwind, negative for headwind

        d_alt, _, sigma, delta, _, c = atmos(alt, isa_diff)
        _, _, drag, _, _, vktas, v_true_fps, thrust, drag_gnd, vkias, m = physics(
            t_inc,
            gamma,
            sigma,
            delta,
            w,
            m,
            c,
            vkias,
            roc_fpm,
            roc_goal,
            speed_goal,
            thrust_factor,
            engines,
            d_alt,
            thrust_mult,
            cdo,
            dcdo_flap1,
            dcdo_flap2,
            dcdo_flap3,
            dcdo_gear,
            k,
            s,
            segment,
            mu_lnd,
            mu_to,
            climb_trigger,
            p,
            mmo,
            v_true_fps,
        )

        if segment in (0, 6, 7, 13):
            tx = 0
            gamma = 0
        elif segment in (8, 9, 10, 11, 12):
            gamma = roc_goal / 60 / v_true_fps * (6076.12 / 3600)
            if round(vkias) >= round(speed_goal) and thrust > drag:
                if w * sin(gamma) + drag < 100:
                    thrust = 100
                else:
                    thrust = w * sin(gamma) + drag
        else:
            tx = (thrust - drag) / w
            if tx > 1:
                tx = 1
            elif tx < -1:
                tx = -1
            gamma = np.arcsin(tx)

        roc_fps = gamma * v_true_fps / (6076.12 / 3600)
        roc_fpm = roc_fps * 60
        gradient = tan(gamma) * 100 if gamma != 0 else 0  # Calculate gradient at each step

        if segment in (3, 4, 5) and alt > alt_goal:
            segment = 6

        if alt - alt_to >= 35 and segment in (1, 2) and to_flag == 0:
            to_flag = 1
            dist_to_35ft = dist_ft  # Distance from start to 35 ft
            segment1_gradient = tan(gamma) * 100 if gamma != 0 else 0  # Gradient for segment 1

        if alt - alt_to >= 400 and segment == 2:
            segment = 3
            dist_to_400ft = dist_ft  # Distance from start to 400 ft
            segment2_gradient = tan(gamma) * 100 if gamma != 0 else 0  # Gradient for segment 2

        if alt - alt_to >= 1500 and segment == 3 and v1_cut == 1:
            segment = 6
            dist_to_1500ft = dist_ft  # Distance from start to 1500 ft
            segment3_gradient = tan(gamma) * 100 if gamma != 0 else 0  # Gradient for segment 3

        if alt - alt_to >= 1500 and segment == 3 and v1_cut == 0:
            segment = 4
            dist_to_1500ft = dist_ft  # Distance from start to 1500 ft
            segment3_gradient = tan(gamma) * 100 if gamma != 0 else 0  # Gradient for segment 3
            climb_fuel = fuel_burned

        if segment in (4, 5) and roc_fpm < 500:
            if segment == 4:
                speed_goal = v_climb
            elif segment == 5:
                speed_goal = m_climb
            if abs(alt - next_step_alt) < alt_tolerance or next_step_alt == 0:
                next_step_alt = next_step_altitude(alt, alt_goal, next_step_alt)
                predicted_roc_value = predict_roc(
                    next_step_alt,
                    alt,
                    w,
                    m,
                    thrust,
                    drag,
                    vktas,
                    thrust_mult,
                    engines,
                    thrust_factor,
                    cdo,
                    dcdo_flap1,
                    dcdo_flap2,
                    dcdo_flap3,
                    dcdo_gear,
                    k,
                    s,
                    isa_diff,
                    speed_goal,
                    segment,
                    alt_goal,
                )
                if predicted_roc_value <= roc_min or (roc_fpm < roc_min and abs(alt - next_step_alt) < alt_tolerance):
                    segment = 6
                if alt >= alt_goal:
                    segment = 6
            elif roc_fpm < roc_min and predicted_roc_value < roc_min:
                segment = 6
        elif segment in (6, 7) and alt < alt_goal:
            next_step_alt = next_step_altitude(alt, alt_goal, next_step_alt)
            predicted_roc_value = predict_roc(
                next_step_alt,
                alt,
                w,
                m,
                thrust,
                drag,
                vktas,
                thrust_mult,
                engines,
                thrust_factor,
                cdo,
                dcdo_flap1,
                dcdo_flap2,
                dcdo_flap3,
                dcdo_gear,
                k,
                s,
                isa_diff,
                speed_goal,
                segment,
                alt_goal,
            )
            if predicted_roc_value > roc_min and alt < alt_goal:
                segment = 5
            elif m >= m_cruise or abs(thrust - drag) < 1:
                segment = 7

        if segment == 6 and last_segment != 6:
            leveloff += 1
            climb_fuel = fuel_burned
            climb_time = t
            climb_fuel_flag = 1
            climb_dist = dist_ft / 6076.12
            # Record the step altitude if not a glitch
            if (p - last_segment_6_step) > debounce_steps:
                step_altitudes.append(round(alt, 0))
                last_segment_6_step = p
        if first_level_off_alt is None and segment == 6:  # Only set in cruise segment
            first_level_off_alt = alt
            final_results['First Level-Off Alt (ft)'] = round(first_level_off_alt)
            final_results['Cruise - First Level-Off Alt (ft)'] = round(first_level_off_alt)

        v_true_fps_wind = wind_component * 6076.12 / 3600  # Convert knots to ft/s
        dist_ft += (v_true_fps + v_true_fps_wind) * t_inc
        remaining_dist = total_dist - dist_ft / 6076.12
        alt += roc_fps * t_inc
        t += t_inc
        fuel_burned_inc = thrust * sfc * t_inc / 3600
        fuel_burned += fuel_burned_inc
        mission_fuel_remain -= fuel_burned_inc
        w -= fuel_burned_inc
        fob = fuel_start - fuel_burned - taxi_fuel
        fuel_burn_history.append(fuel_burned)

        time_data.append(t / 3600)
        alt_data.append(alt)
        dist_data.append(dist_ft / 6076.12)
        vktas_data.append(vktas)
        vkias_data.append(vkias)
        roc_data.append(roc_fpm)
        thrust_data.append(thrust)
        drag_data.append(drag + drag_gnd)
        segment_data.append(segment)
        mach_data.append(m)
        gradient_data.append(gradient)

        # Track segment start weights and calculate fuel remaining at each phase
        if segment == 0 and takeoff_start_weight == 0:  # Start of takeoff
            takeoff_start_weight = w
            # Calculate fuel remaining at start of takeoff
            fuel_remaining = initial_fuel - taxi_fuel
        elif segment == 1 and climb_start_weight == 0:  # Start of climb
            climb_start_weight = w
            takeoff_end_weight = w  # End of takeoff is start of climb
            # Calculate fuel remaining at end of takeoff
            fuel_remaining = initial_fuel - taxi_fuel - fuel_burned
        elif segment == 6 and cruise_start_weight == 0:  # Start of cruise
            cruise_start_weight = w
            climb_end_weight = w  # End of climb is start of cruise
            # Calculate fuel remaining at end of climb
            fuel_remaining = initial_fuel - taxi_fuel - fuel_burned
        elif segment == 8 and descent_start_weight == 0:  # Start of descent
            descent_start_weight = w
            cruise_end_weight = w  # End of cruise is start of descent
            # Calculate fuel remaining at end of cruise
            fuel_remaining = initial_fuel - taxi_fuel - fuel_burned
        elif segment == 12 and landing_start_weight == 0:  # Start of landing
            landing_start_weight = w
            descent_end_weight = w  # End of descent is start of landing
            # Calculate fuel remaining at end of descent
            fuel_remaining = initial_fuel - taxi_fuel - fuel_burned
        
        # Track end of landing phase
        if segment == 14 and landing_end_weight == 0:
            landing_end_weight = w
            # Calculate final fuel remaining
            fuel_remaining = initial_fuel - taxi_fuel - fuel_burned

        # Track descent start
        if segment == 8 and descent_start_time == 0:
            descent_start_time = t
            descent_start_fuel = fuel_burned
            descent_start_dist = dist_ft
            descent_start_alt = alt

        # Track landing start
        if segment == 12 and landing_start_time == 0:
            landing_start_time = t
            landing_start_fuel = fuel_burned
            landing_start_dist = dist_ft
            landing_start_alt = alt

        # Track landing distances
        if segment == 12:  # Landing approach phase
            if alt - alt_land <= 35 and landing_start_time == 0:
                landing_start_time = t
                landing_start_dist = dist_ft
                dist_land_35 = dist_ft
            if alt <= alt_land and landing_start_time > 0:
                dist_land = dist_ft
                dist_ground_roll = dist_land - dist_land_35

        p += 1
        last_segment = segment

        if vkias >= vr and segment == 0:
            segment = 1
            takeoff_roll_dist = dist_ft  # Capture distance at end of segment 0
            if v1_cut == 1:
                engines = 1  # Simulate single-engine operation for V1 cut
        if alt - alt_to >= 35 and segment in (1, 2) and to_flag == 0:
            to_flag = 1
            dist_to_35ft = dist_ft  # Distance from start to 35 ft
            segment1_gradient = tan(gamma) * 100 if gamma != 0 else 0  # Gradient for segment 1
        if segment == 6 and climb_fuel_flag == 0:
            climb_fuel = fuel_burned
            climb_time = t
            climb_fuel_flag = 1
            climb_dist = dist_ft / 6076.12
        if vkias >= v1 and segment == 1:
            segment = 2
        if alt - alt_to >= 400 and segment == 2:
            segment = 3
            dist_to_400ft = dist_ft  # Distance from start to 400 ft
            segment2_gradient = tan(gamma) * 100 if gamma != 0 else 0  # Gradient for segment 2
        if alt - alt_to >= 1500 and segment == 3 and v1_cut == 1:
            segment = 6
            dist_to_1500ft = dist_ft  # Distance from start to 1500 ft
            segment3_gradient = tan(gamma) * 100 if gamma != 0 else 0  # Gradient for segment 3
        if alt - alt_to >= 1500 and segment == 3 and v1_cut == 0:
            segment = 4
            dist_to_1500ft = dist_ft  # Distance from start to 1500 ft
            segment3_gradient = tan(gamma) * 100 if gamma != 0 else 0  # Gradient for segment 3
            climb_fuel = fuel_burned
        if m >= m_climb and segment == 4:
            segment = 5
        if (abs(round(thrust - drag)) < 1 or m >= m_cruise or m >= mmo) and segment == 6:
            segment = 7
        if segment in (6, 7) and remaining_dist <= descent_threshold and alt > 10000:
            segment = 8
            gamma = -0.01
            fuel_start_descent = fob
            t_start_descent = t
            cruise_time = t - climb_time
            cruise_dist = (dist_ft / 6076.12) - climb_dist
            cruise_fuel = fuel_burned - climb_fuel
            max_m_reached = m
        if vkias > v_descent and segment == 8:
            segment = 9
        if alt < 10000 and segment in (8, 9):
            segment = 10
        if alt - alt_land <= 3000 and segment == 10:
            segment = 11
        if alt - alt_land <= 1000 and segment == 11:
            segment = 12
        if (alt - alt_land) <= 35 and segment == 12 and land_trigger == 0:
            dist_at_35 = dist_ft
            land_trigger = 1
        if alt <= alt_land and segment == 12:
            segment = 13
            dist_touchdown = dist_ft
        if segment == 13 and vkias <= 0:
            dist_land = dist_ft - dist_touchdown
            segment = 14
        if vkias <= 0 and segment == 14:
            dist_land = dist_ft - dist_touchdown
            segment = 14
            # Ensure landing distances are valid
            if dist_land_35 <= 0:
                dist_land_35 = 0  # Can't have negative distance

            # Store landing distances in results dictionary
            final_results["Landing - Dist from 35 ft to Stop (ft)"] = int(dist_land_35)
            final_results["Landing - Ground Roll (ft)"] = int(dist_land)

    # Calculate final descent and landing metrics
    descent_time = (t - descent_start_time) / 60 if descent_start_time > 0 else 0
    descent_dist = (dist_ft - descent_start_dist) / 6076.12 if descent_start_dist > 0 else 0
    descent_fuel = fuel_burned - descent_start_fuel if descent_start_fuel > 0 else 0
    
    # Calculate fuel remaining at end of each phase
    takeoff_fuel_remaining = initial_fuel - taxi_fuel - (0 if takeoff_end_weight == 0 else takeoff_start_weight - takeoff_end_weight)
    climb_fuel_remaining = initial_fuel - taxi_fuel - (0 if climb_end_weight == 0 else takeoff_end_weight - climb_end_weight)
    cruise_fuel_remaining = initial_fuel - taxi_fuel - (0 if cruise_end_weight == 0 else cruise_start_weight - cruise_end_weight)
    descent_fuel_remaining = initial_fuel - taxi_fuel - (0 if descent_end_weight == 0 else descent_start_weight - descent_end_weight)
    landing_fuel_remaining = initial_fuel - taxi_fuel - fuel_burned
    
    # Collect final results
    final_results.update({
        # Takeoff section
        "Takeoff Roll Dist (ft)": int(takeoff_roll_dist) if takeoff_roll_dist > 0 else None,
        "Takeoff Start Weight (lb)": int(takeoff_start_weight) if takeoff_start_weight > 0 else None,
        "Takeoff End Weight (lb)": int(takeoff_end_weight) if takeoff_end_weight > 0 else None,
        "Fuel Remaining After Takeoff (lb)": int(takeoff_fuel_remaining) if takeoff_fuel_remaining > 0 else None,
        
        # Climb section
        "Dist to 35 ft (ft)": int(dist_to_35ft) if dist_to_35ft > 0 else None,
        "Segment 1 Gradient (%)": round(segment1_gradient, 2) if segment1_gradient != 0 else None,
        "Dist to 400 ft (ft)": int(dist_to_400ft) if dist_to_400ft > 0 else None,
        "Segment 2 Gradient (%)": round(segment2_gradient, 2) if segment2_gradient != 0 else None,
        "Dist to 1500 ft (ft)": int(dist_to_1500ft) if dist_to_1500ft > 0 else None,
        "Segment 3 Gradient (%)": round(segment3_gradient, 2) if segment3_gradient != 0 else None,
        "Climb Time (min)": int(climb_time / 60) if climb_time > 0 else None,
        "Climb Dist (NM)": int(climb_dist) if climb_dist > 0 else None,
        "Climb Start Weight (lb)": int(climb_start_weight) if climb_start_weight > 0 else None,
        "Climb End Weight (lb)": int(climb_end_weight) if climb_end_weight > 0 else None,
        "Climb Fuel (lb)": int(climb_fuel) if climb_fuel > 0 else None,
        "Fuel Remaining After Climb (lb)": int(climb_fuel_remaining) if climb_fuel_remaining > 0 else None,
        
        # Cruise section
        "Cruise Time (min)": int(cruise_time / 60) if cruise_time > 0 else None,
        "Cruise Dist (NM)": int(cruise_dist) if cruise_dist > 0 else None,
        "Cruise Start Weight (lb)": int(cruise_start_weight) if cruise_start_weight > 0 else None,
        "Cruise End Weight (lb)": int(cruise_end_weight) if cruise_end_weight > 0 else None,
        "Cruise Fuel (lb)": int(cruise_fuel) if cruise_fuel > 0 else None,
        "Cruise VKTAS (knots)": int(max_m_reached * 661.48) if max_m_reached > 0 else None,
        "Cruise - First Level-Off Alt (ft)": int(first_level_off_alt) if first_level_off_alt is not None else None,
        "Step Altitudes (ft)": step_altitudes if step_altitudes else None,
        "Fuel Remaining After Cruise (lb)": int(cruise_fuel_remaining) if cruise_fuel_remaining > 0 else None,
        
        # Descent section
        "Descent Time (min)": int(descent_time) if descent_time > 0 else None,
        "Descent Dist (NM)": int(descent_dist) if descent_dist > 0 else None,
        "Descent Start Weight (lb)": int(descent_start_weight) if descent_start_weight > 0 else None,
        "Descent End Weight (lb)": int(descent_end_weight) if descent_end_weight > 0 else None,
        "Descent Fuel (lb)": int(descent_fuel) if descent_fuel > 0 else None,
        "Fuel Remaining After Descent (lb)": int(descent_fuel_remaining) if descent_fuel_remaining > 0 else None,
        
        # Landing section
        "Landing - Dist from 35 ft to Stop (ft)": int(dist_land_35) if dist_land > 0 else None,
        "Landing - Ground Roll (ft)": int(dist_ground_roll) if dist_ground_roll > 0 else None,
        "Landing Start Weight (lb)": int(landing_start_weight) if landing_start_weight > 0 else None,
        "Landing End Weight (lb)": int(landing_end_weight) if landing_end_weight > 0 else None,
        "Fuel Remaining After Landing (lb)": int(landing_fuel_remaining) if landing_fuel_remaining > 0 else None,
        
        # Totals
        "Total Time (min)": int(t / 60) if t > 0 else None,
        "Total Dist (NM)": int(dist_ft / 6076.12) if dist_ft > 0 else None,
        "Total Fuel Burned (lb)": int(fuel_burned) if fuel_burned > 0 else None,
        "Fuel Remaining (lb)": int(landing_fuel_remaining) if landing_fuel_remaining > 0 else None,
        
        # Other data
        "V1 Cut": v1_cut_enabled,
        "First Level-Off Alt (ft)": int(first_level_off_alt) if first_level_off_alt is not None else None,
    })

    # If V1 cut was enabled, skip landing calculations
    if v1_cut == 0:
        # Add landing distances if the simulation completed a landing
        if dist_land > 0:
            final_results["Landing - Dist from 35 ft to Stop (ft)"] = int(dist_land_35)
            final_results["Landing - Ground Roll (ft)"] = int(dist_land)

    # Create fuel vs distance plot using plotly.graph_objects
    fig = go.Figure()
    
    # Add main line
    fig.add_trace(go.Scatter(
        x=dist_data,  # Already in NM
        y=[initial_fuel - f for f in fuel_burn_history],
        mode='lines',
        name='Fuel Remaining',
        line=dict(color='blue', width=2)
    ))
    
    # Add segment markers
    if climb_dist > 0:
        fig.add_vline(
            x=climb_dist,
            line_dash="dash",
            line_color="green",
            annotation_text="End of Climb",
            annotation_position="top right"
        )
    if cruise_dist > 0:
        fig.add_vline(
            x=climb_dist + cruise_dist,
            line_dash="dash",
            line_color="blue",
            annotation_text="Start of Descent",
            annotation_position="top right"
        )
    if descent_dist > 0:
        fig.add_vline(
            x=climb_dist + cruise_dist + descent_dist,
            line_dash="dash",
            line_color="red",
            annotation_text="End of Descent",
            annotation_position="top right"
        )
    
    # Update layout
    fig.update_layout(
        title="Fuel Remaining vs Distance",
        xaxis_title="Distance (NM)",
        yaxis_title="Fuel Remaining (lb)",
        showlegend=True
    )
    
    # Debug: Log final_results before returning
    print("\n=== Final Results from run_simulation ===")
    print(f"Takeoff V-Speeds: {final_results['Takeoff V-Speeds']}")
    print(f"Approach V-Speeds: {final_results['Approach V-Speeds']}")
    print(f"Total Distance (NM): {int(dist_ft / 6076.12)}")

    # Create the results DataFrame
    results_df = pd.DataFrame({
        'Time (hr)': time_data,
        'Altitude (ft)': alt_data,
        'Distance (NM)': dist_data,
        'VKTAS (kts)': vktas_data,
        'VKIAS (kts)': vkias_data,
        'ROC (fpm)': roc_data,
        'Thrust (lb)': thrust_data,
        'Drag (lb)': drag_data,
        'Segment': segment_data,
        'Mach': mach_data,
        'Gradient (%)': gradient_data,
    })
    
    return results_df, final_results, dep_latitude, dep_longitude, arr_latitude, arr_longitude