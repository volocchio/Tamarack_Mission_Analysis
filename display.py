import streamlit as st
import plotly.graph_objects as go

def display_vspeeds(label: str, vspeeds_data: dict, phase: str = "Takeoff"):
    """Display V-speeds for a given flight phase.
    
    Args:
        label: Label for the display section
        vspeeds_data: Dictionary containing V-speeds data
        phase: Flight phase (default: "Takeoff")
    """
    print(f"\n=== DEBUG: display_vspeeds called for {phase} ===")
    print(f"V-speeds data received: {vspeeds_data}")
    
    # Handle case where vspeeds_data is None or not a dictionary
    if not vspeeds_data or not isinstance(vspeeds_data, dict):
        print(f"Error: Invalid V-speeds data for {phase}: {vspeeds_data}")
        st.warning(f"No valid V-speeds data available for {phase}")
        return
    
    # Display the V-speeds in a clean format
    st.markdown(f"**{phase} V-Speeds**")
    
    # Display weight if available
    weight = vspeeds_data.get('Weight') or vspeeds_data.get('weight')
    if weight is not None:
        st.write(f"Weight: {weight} lb")
    
    # Define the keys we want to display based on phase
    if phase.lower() == 'takeoff':
        keys_to_display = {
            'VR': 'VR',
            'V1': 'V1',
            'V2': 'V2',
            'V3': 'V3'
        }
    else:  # Approach/Landing
        keys_to_display = {
            'VAPP': 'VAPP',
            'VREF': 'VREF'
        }
    
    # Display each V-speed with its value
    any_displayed = False
    for display_key, data_key in keys_to_display.items():
        # Try uppercase, lowercase, and title case versions of the key
        value = vspeeds_data.get(data_key) or vspeeds_data.get(data_key.lower()) or vspeeds_data.get(data_key.title())
        if value is not None:
            # Format the value to 1 decimal place if it's a float
            if isinstance(value, (float, int)) and not isinstance(value, bool):
                value = f"{float(value):.1f}"
            st.write(f"{display_key}: {value} kts")
            any_displayed = True
    
    # If no V-speeds were displayed, show a warning with available keys
    if not any_displayed:
        error_msg = f"No valid V-speeds found for {phase}. "
        error_msg += f"Available keys: {list(vspeeds_data.keys())}"
        print(error_msg)
        st.warning(f"No valid V-speeds found for {phase}")
    else:
        print(f"Successfully displayed V-speeds for {phase}")

def write_metrics_with_headings(results_dict, label):
    print(f"\n=== DEBUG: write_metrics_with_headings called for {label} ===")
    print(f"All keys in results_dict: {list(results_dict.keys())}")
    
    # Debug: Print V-speeds keys specifically
    vspeed_keys = [k for k in results_dict.keys() if "V-Speeds" in k or "VSpeed" in k]
    print(f"V-speeds related keys: {vspeed_keys}")
    
    segment_definitions = {
        "Takeoff": {
            "start": "Takeoff Start Weight (lb)",
            "end": "Takeoff End Weight (lb)",
            "vspeeds": True,
            "vspeeds_phase": "Takeoff",
            "fields": [
                "Takeoff Roll Dist (ft)", "Dist to 35 ft (ft)",
                "Segment 1 Gradient (%)", "Dist to 400 ft (ft)", "Segment 2 Gradient (%)", "Dist to 1500 ft (ft)", "Segment 3 Gradient (%)",
                "Climb Fuel (lb)", "Fuel Remaining After Takeoff (lb)"
            ]
        },
        "Climb": {
            "start": "Climb Start Weight (lb)",
            "end": "Climb End Weight (lb)",
            "fields": ["Climb Time (min)", "Climb Dist (NM)", "Climb Fuel (lb)", "Fuel Remaining After Climb (lb)"]
        },
        "Cruise": {
            "start": "Cruise Start Weight (lb)",
            "end": "Cruise End Weight (lb)",
            "fields": [
                "Cruise Time (min)", "Cruise Dist (NM)", "Cruise Fuel (lb)", "Cruise VKTAS (knots)",
                "Cruise - First Level-Off Alt (ft)", "Fuel Remaining After Cruise (lb)"
            ]
        },
        "Descent": {
            "start": "Descent Start Weight (lb)",
            "end": "Descent End Weight (lb)",
            "fields": ["Descent Time (min)", "Descent Dist (NM)", "Descent Fuel (lb)", "Fuel Remaining After Descent (lb)"]
        },
        "Landing": {
            "start": "Landing Start Weight (lb)",
            "end": "Landing End Weight (lb)",
            "vspeeds": True,
            "vspeeds_phase": "Approach",
            "fields": ["Landing - Dist from 35 ft to Stop (ft)", "Landing - Ground Roll (ft)", "Fuel Remaining After Landing (lb)"]
        },
        "Total": {
            "start": None,
            "end": None,
            "fields": ["Total Time (min)", "Total Dist (NM)", "Total Fuel Burned (lb)", "Fuel Remaining (lb)", "V1 Cut"]
        }
    }

    for section, config in segment_definitions.items():
        st.markdown(f"### {section}")
        if config.get("start") and config["start"] in results_dict:
            st.write(f"Start Weight: {results_dict[config['start']]} lb")
        if config.get("end") and config["end"] in results_dict:
            st.write(f"End Weight: {results_dict[config['end']]} lb")

        if config.get("vspeeds"):
            phase = config.get("vspeeds_phase", section)
            print(f"\n--- Looking for V-speeds for {phase} ---")
            print(f"Section: {section}, Phase: {phase}")
            
            # Use the primary key
            vspeeds_key = f"{phase} V-Speeds"
            
            print(f"Looking for V-speeds with key: '{vspeeds_key}'")
            print(f"Available keys: {list(results_dict.keys())}")
            
            if vspeeds_key in results_dict and results_dict[vspeeds_key] is not None:
                vspeeds_data = results_dict[vspeeds_key]
                print(f"Found V-speeds data: {vspeeds_data}")
                
                # Display the V-speeds
                display_vspeeds(label, vspeeds_data, phase)
            else:
                print(f"WARNING: No V-speeds found in results_dict for phase {phase}")
                st.warning(f"V-speeds data not found for {phase}")

        for field in config.get("fields", []):
            if field in results_dict:
                st.write(f"{field}: {results_dict[field]}")

def plot_flight_profiles(tamarack_data, flatwing_data):
    st.subheader("Flight Profile Charts")

    def plot_dual_y(title, x, y1_label, y1_tam, y1_flat, y2_label, y2_tam, y2_flat):
        fig = go.Figure()
        if not tamarack_data.empty:
            fig.add_trace(go.Scatter(x=tamarack_data[x], y=tamarack_data[y1_tam],
                                     name=f"{y1_label} (Tamarack)", line=dict(color="blue")))
            fig.add_trace(go.Scatter(x=tamarack_data[x], y=tamarack_data[y2_tam],
                                     name=f"{y2_label} (Tamarack)", yaxis='y2', line=dict(color="orange")))
        if not flatwing_data.empty:
            fig.add_trace(go.Scatter(x=flatwing_data[x], y=flatwing_data[y1_flat],
                                     name=f"{y1_label} (Flatwing)", line=dict(color="purple")))
            fig.add_trace(go.Scatter(x=flatwing_data[x], y=flatwing_data[y2_flat],
                                     name=f"{y2_label} (Flatwing)", yaxis='y2', line=dict(color="red")))

        fig.update_layout(
            xaxis=dict(title=x),
            yaxis=dict(title=y1_label),
            yaxis2=dict(title=y2_label, overlaying='y', side='right'),
            title=title,
            legend=dict(x=0.01, y=1.15, orientation="h")
        )
        st.plotly_chart(fig, use_container_width=True)

    def plot_single_y(title, x, y_label, y_tam, y_flat):
        fig = go.Figure()
        if not tamarack_data.empty and y_tam in tamarack_data.columns:
            fig.add_trace(go.Scatter(x=tamarack_data[x], y=tamarack_data[y_tam],
                                     name=f"{y_label} (Tamarack)", line=dict(color="blue")))
        if not flatwing_data.empty and y_flat in flatwing_data.columns:
            fig.add_trace(go.Scatter(x=flatwing_data[x], y=flatwing_data[y_flat],
                                     name=f"{y_label} (Flatwing)", line=dict(color="purple")))

        if not fig.data:
            st.warning(f"Skipping '{title}' – column missing in both datasets.")
            return

        fig.update_layout(
            xaxis=dict(title=x),
            yaxis=dict(title=y_label),
            title=title,
            legend=dict(x=0.01, y=1.15, orientation="h")
        )
        st.plotly_chart(fig, use_container_width=True)

    plot_dual_y("Altitude and Mach vs. Distance", "Distance (NM)",
                "Altitude (ft)", "Altitude (ft)", "Altitude (ft)",
                "Mach", "Mach", "Mach")

    for title in [
        ("Rate of Climb vs. Distance", "ROC (fpm)"),
        ("Thrust vs. Distance", "Thrust (lb)"),
        ("Drag vs. Distance", "Drag (lb)")
    ]:
        plot_single_y(title[0], "Distance (NM)", title[1], title[1], title[1])

    st.subheader("Fuel Remaining vs Distance")
    if "fuel_distance_plot" in tamarack_data:
        st.plotly_chart(tamarack_data["fuel_distance_plot"], use_container_width=True)
    elif "fuel_distance_plot" in flatwing_data:
        st.plotly_chart(flatwing_data["fuel_distance_plot"], use_container_width=True)

def display_simulation_results(
    tamarack_data, tamarack_results, flatwing_data, flatwing_results, v1_cut_enabled,
    dep_latitude, dep_longitude, arr_latitude, arr_longitude, distance_nm, bearing_deg,
    winds_temps_source, cruise_alt, departure_airport, arrival_airport, initial_fuel
):
    # Debug: Print all keys in the results dictionaries
    print("\n=== DEBUG: Tamarack Results Keys ===")
    print("\n".join(tamarack_results.keys()))
    print("\n=== DEBUG: Flatwing Results Keys ===")
    print("\n".join(flatwing_results.keys()))
    
    # Debug: Print V-speeds specifically
    for config_name, results in [("Tamarack", tamarack_results), ("Flatwing", flatwing_results)]:
        print(f"\n=== DEBUG: {config_name} V-Speeds ===")
        for key in ["Takeoff V-Speeds", "Approach V-Speeds"]:
            if key in results:
                print(f"{key}: {results[key]}")
            else:
                print(f"{key}: NOT FOUND")
    st.subheader("Flight Route Map")
    fig = go.Figure()

    fig.add_trace(go.Scattergeo(
        lat=[dep_latitude, arr_latitude],
        lon=[dep_longitude, arr_longitude],
        mode='lines',
        line=dict(width=2, color='blue'),
        name='Route'
    ))
    fig.add_trace(go.Scattergeo(
        lat=[dep_latitude],
        lon=[dep_longitude],
        mode='markers',
        marker=dict(size=10, color='green'),
        name='Departure'
    ))
    fig.add_trace(go.Scattergeo(
        lat=[arr_latitude],
        lon=[arr_longitude],
        mode='markers',
        marker=dict(size=10, color='red'),
        name='Arrival'
    ))
    fig.update_layout(
        geo=dict(
            scope='north america',
            showland=True,
            landcolor='rgb(243, 243, 243)',
            countrycolor='rgb(204, 204, 204)',
            showcountries=True,
            projection_type='equirectangular',
            center=dict(lat=(dep_latitude + arr_latitude) / 2,
                        lon=(dep_longitude + arr_longitude) / 2),
            lataxis=dict(range=[min(dep_latitude, arr_latitude) - 5, max(dep_latitude, arr_latitude) + 5]),
            lonaxis=dict(range=[min(dep_longitude, arr_longitude) - 10, max(dep_longitude, arr_longitude) + 10])
        ),
        title='Flight Route'
    )
    st.plotly_chart(fig, use_container_width=True)
    st.success(f"Distance: {distance_nm:.1f} NM | Bearing: {bearing_deg:.1f}°")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Tamarack")
        if not tamarack_data.empty:
            write_metrics_with_headings(tamarack_results, "Tamarack")
        else:
            st.info("No Tamarack results to display.")

    with col2:
        st.subheader("Flatwing")
        if not flatwing_data.empty:
            write_metrics_with_headings(flatwing_results, "Flatwing")
        else:
            st.info("No Flatwing results to display.")

    for results in [tamarack_results, flatwing_results]:
        if results.get("exceedances"):
            for msg in results["exceedances"]:
                st.error(msg)
        elif results.get("error"):
            st.error(results["error"])

    plot_flight_profiles(tamarack_data, flatwing_data)