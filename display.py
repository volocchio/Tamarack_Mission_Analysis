import streamlit as st
import plotly.graph_objects as go
import plotly.io as pio
from io import BytesIO
import datetime
import os
import numpy as np
import pandas as pd
from simulation import get_global_timestamp
from utils import load_airports
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

def display_vspeeds(label: str, vspeeds_data: dict, phase: str = "Takeoff"):
    """Display V-speeds for a given flight phase.
    
    Args:
        label: Label for the display section
        vspeeds_data: Dictionary containing V-speeds data
        phase: Flight phase (default: "Takeoff")
    """
    
    # Handle case where vspeeds_data is None or not a dictionary
    if not vspeeds_data or not isinstance(vspeeds_data, dict):
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
        st.warning(f"No valid V-speeds found for {phase}")
    else:
        pass

def write_metrics_with_headings(results_dict, label):
    
    
    segment_definitions = {
        "Takeoff": {
            "start": "Takeoff Start Weight (lb)",
            "end": "Takeoff End Weight (lb)",
            "vspeeds": True,
            "vspeeds_phase": "Takeoff",
            "fields": [
                "Takeoff Roll Dist (ft)", "Dist to 35 ft (ft)",
                "Segment 1 Gradient (%)", "Dist to 400 ft (ft)", "Segment 2 Gradient (%)", "Dist to 1500 ft (ft)", "Segment 3 Gradient (%)",
                "Takeoff Fuel (lb)", "Fuel Remaining After Takeoff (lb)"
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
                "Cruise Max Mach", "Cruise - First Level-Off Alt (ft)", "Fuel Remaining After Cruise (lb)"
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
            
            # Use the primary key
            vspeeds_key = f"{phase} V-Speeds"
            
            
            if vspeeds_key in results_dict and results_dict[vspeeds_key] is not None:
                vspeeds_data = results_dict[vspeeds_key]
                
                # Display the V-speeds
                display_vspeeds(label, vspeeds_data, phase)
            else:
                st.warning(f"V-speeds data not found for {phase}")

        for field in config.get("fields", []):
            # Special handling: show 'Takeoff Fuel (lb)' in Takeoff section
            if section == "Takeoff" and field == "Takeoff Fuel (lb)":
                val = None
                try:
                    sw = results_dict.get("Takeoff Start Weight (lb)")
                    ew = results_dict.get("Takeoff End Weight (lb)")
                    if sw is not None and ew is not None:
                        val = int(sw) - int(ew)
                except Exception:
                    val = None
                if val is not None and val >= 0:
                    st.write(f"Takeoff Fuel (lb): {val}")
                continue
            if field in results_dict:
                st.write(f"{field}: {results_dict[field]}")
        # Add Cruise Avg PPH (lb/hr) under Cruise section
        if section == "Cruise":
            try:
                burn_lb = results_dict.get("Cruise Fuel (lb)")
                time_min = results_dict.get("Cruise Time (min)")
                if burn_lb is not None and time_min is not None:
                    tmin = float(time_min)
                    if tmin > 0:
                        pph = float(burn_lb) / (tmin / 60.0)
                        st.write(f"Cruise Avg PPH (lb/hr): {int(round(pph))}")
            except Exception:
                pass
        # Add PPH/GPH and block speed under Total section
        if section == "Total":
            try:
                burn_lb = results_dict.get("Total Fuel Burned (lb)")
                time_min = results_dict.get("Total Time (min)")
                dist_nm = results_dict.get("Total Dist (NM)")
                if burn_lb is not None and time_min is not None:
                    tmin = float(time_min)
                    if tmin > 0:
                        pph = float(burn_lb) / (tmin / 60.0)
                        gph = pph / 6.7
                        st.write(f"PPH (lb/hr): {int(round(pph))} | GPH (gal/hr): {gph:.1f}")
                if time_min is not None and dist_nm is not None:
                    tmin = float(time_min)
                    dnm = float(dist_nm)
                    if tmin > 0:
                        hours = tmin / 60.0
                        block_speed = dnm / hours
                        st.write(f"Block Speed (kt): {block_speed:.1f}")
            except Exception:
                pass

def plot_flight_profiles(tamarack_data, flatwing_data, tamarack_results, flatwing_results):
    st.subheader("Flight Profile Charts")

    figs = {}

    def plot_dual_y(title, key_name, x, y1_label, y1_tam, y1_flat, y2_label, y2_tam, y2_flat):
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
            xaxis=dict(title=x, showgrid=True, gridcolor='LightGrey', gridwidth=1),
            yaxis=dict(title=y1_label),
            yaxis2=dict(title=y2_label, overlaying='y', side='right'),
            title=title,
            legend=dict(x=0.01, y=1.15, orientation="h")
        )
        st.plotly_chart(fig, use_container_width=True)
        figs[key_name] = fig

    def plot_single_y(title, key_name, x, y_label, y_tam, y_flat):
        fig = go.Figure()
        if not tamarack_data.empty and y_tam in tamarack_data.columns:
            fig.add_trace(go.Scatter(x=tamarack_data[x], y=tamarack_data[y_tam],
                                     name=f"{y_label} (Tamarack)", line=dict(color="blue")))
        if not flatwing_data.empty and y_flat in flatwing_data.columns:
            fig.add_trace(go.Scatter(x=flatwing_data[x], y=flatwing_data[y_flat],
                                     name=f"{y_label} (Flatwing)", line=dict(color="purple")))

        if not fig.data:
            st.warning(f"Skipping '{title}' – column missing in both datasets.")
            return None

        fig.update_layout(
            xaxis=dict(title=x, showgrid=True, gridcolor='LightGrey', gridwidth=1),
            yaxis=dict(title=y_label),
            title=title,
            legend=dict(x=0.01, y=1.15, orientation="h")
        )
        st.plotly_chart(fig, use_container_width=True)
        figs[key_name] = fig
        return fig

    plot_dual_y("Altitude and Mach vs. Distance", "alt_mach", "Distance (NM)",
                "Altitude (ft)", "Altitude (ft)", "Altitude (ft)",
                "Mach", "Mach", "Mach")

    
    fig = go.Figure()
    if not tamarack_data.empty:
        if "Distance (NM)" in tamarack_data.columns and "Altitude (ft)" in tamarack_data.columns:
            fig.add_trace(go.Scatter(x=tamarack_data["Distance (NM)"], y=tamarack_data["Altitude (ft)"],
                                     name="Altitude (Tamarack)", line=dict(color="blue")))
        if "Distance (NM)" in tamarack_data.columns and "VKTAS (kts)" in tamarack_data.columns:
            fig.add_trace(go.Scatter(x=tamarack_data["Distance (NM)"], y=tamarack_data["VKTAS (kts)"],
                                     name="TAS (Tamarack)", yaxis='y2', line=dict(color="orange", dash="solid")))
        if "Distance (NM)" in tamarack_data.columns and "VKIAS (kts)" in tamarack_data.columns:
            fig.add_trace(go.Scatter(x=tamarack_data["Distance (NM)"], y=tamarack_data["VKIAS (kts)"],
                                     name="IAS (Tamarack)", yaxis='y2', line=dict(color="orange", dash="dash")))
    if not flatwing_data.empty:
        if "Distance (NM)" in flatwing_data.columns and "Altitude (ft)" in flatwing_data.columns:
            fig.add_trace(go.Scatter(x=flatwing_data["Distance (NM)"], y=flatwing_data["Altitude (ft)"],
                                     name="Altitude (Flatwing)", line=dict(color="purple")))
        if "Distance (NM)" in flatwing_data.columns and "VKTAS (kts)" in flatwing_data.columns:
            fig.add_trace(go.Scatter(x=flatwing_data["Distance (NM)"], y=flatwing_data["VKTAS (kts)"],
                                     name="TAS (Flatwing)", yaxis='y2', line=dict(color="red", dash="solid")))
        if "Distance (NM)" in flatwing_data.columns and "VKIAS (kts)" in flatwing_data.columns:
            fig.add_trace(go.Scatter(x=flatwing_data["Distance (NM)"], y=flatwing_data["VKIAS (kts)"],
                                     name="IAS (Flatwing)", yaxis='y2', line=dict(color="red", dash="dash")))

    if fig.data:
        fig.update_layout(
            xaxis=dict(title="Distance (NM)", showgrid=True, gridcolor='LightGrey', gridwidth=1),
            yaxis=dict(title="Altitude (ft)"),
            yaxis2=dict(title="Speed (kt)", overlaying='y', side='right'),
            title="Altitude, TAS and IAS vs. Distance",
            legend=dict(x=0.01, y=1.15, orientation="h")
        )
        st.plotly_chart(fig, use_container_width=True)
        figs["alt_tas_ias"] = fig

    for title, col in [
        ("Rate of Climb vs. Distance", "ROC (fpm)"),
        ("Thrust vs. Distance", "Thrust (lb)"),
        ("Drag vs. Distance", "Drag (lb)")
    ]:
        plot_single_y(title, col, "Distance (NM)", col, col, col)

    st.subheader("Fuel Remaining vs Distance")
    
    # If we have both aircraft data, create a combined comparison plot using complete time history
    if not tamarack_data.empty and not flatwing_data.empty:
        # Create combined comparison plot
        fig = go.Figure()
        
        # Add Tamarack data - use complete time history from results_df
        if 'Distance (NM)' in tamarack_data.columns:
            # Get initial fuel from results
            tamarack_initial_fuel = tamarack_results.get('Total Fuel Burned (lb)', 0) + tamarack_results.get('Fuel Remaining (lb)', 0)
            
            # Calculate cumulative fuel burn from complete time history
            tamarack_fuel_remaining = []
            cumulative_fuel = 0
            
            # Use all data points - no skipping
            for i in range(len(tamarack_data)):
                if i == 0:
                    # First point has initial fuel
                    tamarack_fuel_remaining.append(tamarack_initial_fuel)
                else:
                    # Calculate fuel burn between points
                    if 'Thrust (lb)' in tamarack_data.columns and 'Time (hr)' in tamarack_data.columns:
                        # Estimate fuel burn from thrust and time
                        avg_thrust = (tamarack_data['Thrust (lb)'].iloc[i-1] + tamarack_data['Thrust (lb)'].iloc[i]) / 2
                        time_diff = tamarack_data['Time (hr)'].iloc[i] - tamarack_data['Time (hr)'].iloc[i-1]
                        # Use SFC approximation (typical for jet aircraft)
                        sfc = 0.7  # lb/hr/lb
                        fuel_burn = avg_thrust * sfc * time_diff
                        cumulative_fuel += fuel_burn
                        tamarack_fuel_remaining.append(tamarack_initial_fuel - cumulative_fuel)
                    else:
                        # Fallback to linear interpolation
                        tamarack_fuel_remaining.append(tamarack_initial_fuel * (1 - i/len(tamarack_data)))
            
            fig.add_trace(go.Scatter(
                x=tamarack_data['Distance (NM)'],
                y=tamarack_fuel_remaining,
                mode='lines',
                name='Tamarack',
                line=dict(color='red', width=2)
            ))
        
        # Add Flatwing data - use complete time history from results_df  
        if 'Distance (NM)' in flatwing_data.columns:
            # Get initial fuel from results
            flatwing_initial_fuel = flatwing_results.get('Total Fuel Burned (lb)', 0) + flatwing_results.get('Fuel Remaining (lb)', 0)
            
            # Calculate cumulative fuel burn from complete time history
            flatwing_fuel_remaining = []
            cumulative_fuel = 0
            
            # Use all data points - no skipping
            for i in range(len(flatwing_data)):
                if i == 0:
                    # First point has initial fuel
                    flatwing_fuel_remaining.append(flatwing_initial_fuel)
                else:
                    # Calculate fuel burn between points
                    if 'Thrust (lb)' in flatwing_data.columns and 'Time (hr)' in flatwing_data.columns:
                        # Estimate fuel burn from thrust and time
                        avg_thrust = (flatwing_data['Thrust (lb)'].iloc[i-1] + flatwing_data['Thrust (lb)'].iloc[i]) / 2
                        time_diff = flatwing_data['Time (hr)'].iloc[i] - flatwing_data['Time (hr)'].iloc[i-1]
                        # Use SFC approximation (typical for jet aircraft)
                        sfc = 0.7  # lb/hr/lb
                        fuel_burn = avg_thrust * sfc * time_diff
                        cumulative_fuel += fuel_burn
                        flatwing_fuel_remaining.append(flatwing_initial_fuel - cumulative_fuel)
                    else:
                        # Fallback to linear interpolation
                        flatwing_fuel_remaining.append(flatwing_initial_fuel * (1 - i/len(flatwing_data)))
            
            fig.add_trace(go.Scatter(
                x=flatwing_data['Distance (NM)'],
                y=flatwing_fuel_remaining,
                mode='lines',
                name='Flatwing',
                line=dict(color='blue', width=2)
            ))
        
        # Update layout for comparison
        fig.update_layout(
            title="Fuel Remaining vs Distance - Comparison",
            xaxis_title="Distance (NM)",
            yaxis_title="Fuel Remaining (lb)",
            showlegend=True,
            xaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='LightGrey',
                showline=True,
                linewidth=1,
                linecolor='Grey',
                mirror=True,
                tickmode='auto',
                nticks=10
            ),
            yaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='LightGrey',
                showline=True,
                linewidth=1,
                linecolor='Grey',
                mirror=True
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        figs["fuel_remaining"] = fig
        
    else:
        # Show single aircraft plot if only one aircraft is available
        if not tamarack_data.empty and "fuel_distance_plot" in tamarack_results:
            st.plotly_chart(tamarack_results["fuel_distance_plot"], use_container_width=True)
            figs["fuel_remaining"] = tamarack_results["fuel_distance_plot"]
        elif not flatwing_data.empty and "fuel_distance_plot" in flatwing_results:
            st.plotly_chart(flatwing_results["fuel_distance_plot"], use_container_width=True)
            figs["fuel_remaining"] = flatwing_results["fuel_distance_plot"]
        return figs

    # Always return figs if combined path was taken
    return figs

def display_simulation_results(
    tamarack_data, tamarack_results, flatwing_data, flatwing_results, v1_cut_enabled,
    dep_latitude, dep_longitude, arr_latitude, arr_longitude, distance_nm, bearing_deg,
    winds_temps_source, isa_dev, cruise_alt, departure_airport, arrival_airport, initial_fuel,
    report_output_dir=None,
    weight_df_flatwing: pd.DataFrame | None = None,
    weight_df_tamarack: pd.DataFrame | None = None,
    weight_df_single: pd.DataFrame | None = None,
    modes_summary_df: pd.DataFrame | None = None,
    fuel_cost_per_gal: float | None = None
):
    # Airports table (restored at top)
    try:
        airports_df = load_airports()
        dep_row = airports_df[airports_df['ident'] == str(departure_airport).upper()].iloc[0]
        arr_row = airports_df[airports_df['ident'] == str(arrival_airport).upper()].iloc[0]

        def short_name(n: str) -> str:
            s = str(n).title()
            s = (s.replace("International", "Intl")
                   .replace("Municipal", "Muni")
                   .replace("Regional", "Rgnl")
                   .replace("County", "Cnty")
                   .replace("Airport", "Arpt"))
            return s

        pa_dep = float(dep_row.get('elevation_ft', 0.0))
        pa_arr = float(arr_row.get('elevation_ft', 0.0))
        da_dep = pa_dep + 120.0 * float(isa_dev)
        da_arr = pa_arr + 120.0 * float(isa_dev)

        airports_table = pd.DataFrame([
            {
                'Type': 'Departure',
                'Airport': short_name(dep_row.get('name', departure_airport)),
                'ICAO': str(departure_airport),
                'PA (ft)': int(round(pa_dep)),
                'DA (ft)': int(round(da_dep)),
            },
            {
                'Type': 'Arrival',
                'Airport': short_name(arr_row.get('name', arrival_airport)),
                'ICAO': str(arrival_airport),
                'PA (ft)': int(round(pa_arr)),
                'DA (ft)': int(round(da_arr)),
            }
        ])
        st.subheader("Airports")
        try:
            st.dataframe(airports_table, use_container_width=True, hide_index=True)
        except TypeError:
            try:
                st.dataframe(airports_table.style.hide(axis='index'), use_container_width=True)
            except Exception:
                st.table(airports_table)
    except Exception:
        pass

    # Weight Status tables (restored at top)
    try:
        has_both = (weight_df_flatwing is not None and isinstance(weight_df_flatwing, pd.DataFrame) and not weight_df_flatwing.empty) \
                   and (weight_df_tamarack is not None and isinstance(weight_df_tamarack, pd.DataFrame) and not weight_df_tamarack.empty)
        has_single = (weight_df_single is not None and isinstance(weight_df_single, pd.DataFrame) and not weight_df_single.empty)
        if has_both or has_single:
            st.subheader('Weight Status')
            def _style_exceed(df: pd.DataFrame):
                try:
                    def _row_style(row):
                        try:
                            w = float(str(row.get('Weight (lb)', '')).replace(',', ''))
                            m = float(str(row.get('Max Weight (lb)', '')).replace(',', ''))
                            return ['background-color: #ffcccc' if w > m else '' for _ in row]
                        except Exception:
                            return ['' for _ in row]
                    return df.style.apply(_row_style, axis=1)
                except Exception:
                    return df
            if has_both:
                col1, col2 = st.columns(2)
                with col1:
                    st.caption("Flatwing")
                    try:
                        st.table(_style_exceed(weight_df_flatwing))
                    except Exception:
                        st.table(weight_df_flatwing)
                with col2:
                    st.caption("Tamarack")
                    try:
                        st.table(_style_exceed(weight_df_tamarack))
                    except Exception:
                        st.table(weight_df_tamarack)
            elif has_single:
                try:
                    st.table(_style_exceed(weight_df_single))
                except Exception:
                    st.table(weight_df_single)
    except Exception:
        pass

    st.subheader("Flight Route Map")
    fig = go.Figure()

    # Add range rings from departure airport using proper equirectangular projection
    import math
    
    def calculate_range_ring_equirectangular(center_lat, center_lon, radius_nm, num_points=72):
        """
        Calculate range ring points using equirectangular projection.
        This matches the map projection for consistent display.
        """
        ring_lats = []
        ring_lons = []
        
        # Convert radius from nautical miles to degrees
        # 1 nautical mile = 1/60 degree of latitude
        radius_deg_lat = radius_nm / 60.0
        
        # For equirectangular projection, longitude scaling depends on latitude
        cos_lat = math.cos(math.radians(center_lat))
        
        for i in range(num_points + 1):  # +1 to close the circle
            angle = 2 * math.pi * i / num_points
            
            # Calculate offsets in the equirectangular projection
            lat_offset = radius_deg_lat * math.cos(angle)
            lon_offset = radius_deg_lat * math.sin(angle) / cos_lat if cos_lat != 0 else 0
            
            ring_lats.append(center_lat + lat_offset)
            ring_lons.append(center_lon + lon_offset)
        
        return ring_lats, ring_lons
    
    # Calculate range ring intervals based on total distance
    max_range = distance_nm * 1.2  # Extend rings 20% beyond destination
    if max_range <= 500:
        ring_interval = 100  # 100 NM intervals for shorter flights
    elif max_range <= 1500:
        ring_interval = 200  # 200 NM intervals for medium flights
    else:
        ring_interval = 500  # 500 NM intervals for long flights
    
    # Add regular interval range rings
    for ring_distance in range(ring_interval, int(max_range) + ring_interval, ring_interval):
        ring_lats, ring_lons = calculate_range_ring_equirectangular(
            dep_latitude, dep_longitude, ring_distance
        )
        
        fig.add_trace(go.Scattergeo(
            lat=ring_lats,
            lon=ring_lons,
            mode='lines',
            line=dict(width=1, color='gray', dash='dot'),
            name=f'{ring_distance} NM',
            showlegend=False,
            hovertemplate=f'Range: {ring_distance} NM<extra></extra>'
        ))
    
    # Add special ring for exact destination distance
    if distance_nm > 0:
        dest_ring_lats, dest_ring_lons = calculate_range_ring_equirectangular(
            dep_latitude, dep_longitude, distance_nm
        )
        
        fig.add_trace(go.Scattergeo(
            lat=dest_ring_lats,
            lon=dest_ring_lons,
            mode='lines',
            line=dict(width=2, color='orange', dash='solid'),
            name=f'Destination: {distance_nm:.1f} NM',
            showlegend=True,
            hovertemplate=f'Destination Range: {distance_nm:.1f} NM<extra></extra>'
        ))

    # Add flight route
    fig.add_trace(go.Scattergeo(
        lat=[dep_latitude, arr_latitude],
        lon=[dep_longitude, arr_longitude],
        mode='lines',
        line=dict(width=3, color='blue'),
        name='Route'
    ))
    
    # Add departure airport
    fig.add_trace(go.Scattergeo(
        lat=[dep_latitude],
        lon=[dep_longitude],
        mode='markers',
        marker=dict(size=12, color='green', symbol='circle'),
        name='Departure',
        text=[departure_airport],
        hovertemplate='%{text}<br>Departure<extra></extra>'
    ))
    
    # Add arrival airport
    fig.add_trace(go.Scattergeo(
        lat=[arr_latitude],
        lon=[arr_longitude],
        mode='markers',
        marker=dict(size=12, color='red', symbol='circle'),
        name='Arrival',
        text=[arrival_airport],
        hovertemplate='%{text}<br>Arrival<extra></extra>'
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

    # Comparison savings after the Total section when both aircraft are present
    try:
        if isinstance(tamarack_results, dict) and isinstance(flatwing_results, dict):
            if tamarack_results.get("Total Fuel Burned (lb)") is not None and flatwing_results.get("Total Fuel Burned (lb)") is not None:
                lb_per_gal = 6.7
                t_burn_lb = float(tamarack_results.get("Total Fuel Burned (lb)", 0) or 0)
                f_burn_lb = float(flatwing_results.get("Total Fuel Burned (lb)", 0) or 0)
                t_time_min = float(tamarack_results.get("Total Time (min)", 0) or 0)
                f_time_min = float(flatwing_results.get("Total Time (min)", 0) or 0)
                t_hours = max(1e-6, t_time_min / 60.0)
                f_hours = max(1e-6, f_time_min / 60.0)
                savings_lb = f_burn_lb - t_burn_lb
                savings_gal = savings_lb / lb_per_gal if lb_per_gal > 0 else 0.0
                t_pph = t_burn_lb / t_hours
                f_pph = f_burn_lb / f_hours
                savings_pph = f_pph - t_pph
                t_cost = None
                f_cost = None
                cost_savings = None
                if fuel_cost_per_gal is not None:
                    t_cost = (t_burn_lb / lb_per_gal) * float(fuel_cost_per_gal)
                    f_cost = (f_burn_lb / lb_per_gal) * float(fuel_cost_per_gal)
                    cost_savings = f_cost - t_cost
                st.subheader("Comparison Savings")
                rows = []
                rows.append({
                    "Metric": "Fuel Saved",
                    "Value": f"{savings_lb:,.0f} lb ({savings_gal:,.1f} gal)"
                })
                rows.append({
                    "Metric": "PPH Saved",
                    "Value": f"{savings_pph:,.0f} lb/hr"
                })
                if t_cost is not None and f_cost is not None and cost_savings is not None:
                    rows.append({"Metric": "Tamarack Fuel Cost", "Value": f"${t_cost:,.0f}"})
                    rows.append({"Metric": "Flatwing Fuel Cost", "Value": f"${f_cost:,.0f}"})
                    rows.append({"Metric": "Cost Savings", "Value": f"${cost_savings:,.0f}"})
                st.table(pd.DataFrame(rows))
    except Exception:
        pass

    for results in [tamarack_results, flatwing_results]:
        if results.get("exceedances"):
            for msg in results["exceedances"]:
                st.error(msg)
        elif results.get("error"):
            st.error(results["error"])

    # Hourly Fuel Burn Summary (moved before Flight Profile Charts)
    st.subheader("Hourly Fuel Burn Summary")
    def _hourly_burns(df: pd.DataFrame) -> list[int]:
        try:
            if df is None or df.empty:
                return []
            if 'Time (hr)' not in df.columns or 'Fuel Remaining (lb)' not in df.columns:
                return []
            t = pd.to_numeric(df['Time (hr)'], errors='coerce')
            f = pd.to_numeric(df['Fuel Remaining (lb)'], errors='coerce')
            mask = t.notna() & f.notna()
            t = t[mask].to_numpy()
            f = f[mask].to_numpy()
            if t.size == 0:
                return []
            t_max = float(np.nanmax(t))
            if t_max <= 0:
                return []
            hours = int(np.ceil(t_max))
            order = np.argsort(t)
            t = t[order]
            f = f[order]
            burns = []
            for h in range(hours):
                start_t = float(h)
                end_t = float(min(h + 1, t_max))
                if end_t <= start_t:
                    continue
                start_f = float(np.interp(start_t, t, f))
                end_f = float(np.interp(end_t, t, f))
                burns.append(int(max(0.0, start_f - end_f)))
            return burns
        except Exception:
            return []
    burns_t = _hourly_burns(tamarack_data)
    burns_f = _hourly_burns(flatwing_data)
    if not burns_t and not burns_f:
        st.info("No time history available to compute hourly burn.")
    else:
        try:
            if burns_t and burns_f:
                n = max(len(burns_t), len(burns_f))
                rows = []
                for i in range(n):
                    rows.append({
                        'Hour': i + 1,
                        'Tamarack (lb)': (burns_t[i] if i < len(burns_t) else None),
                        'Flatwing (lb)': (burns_f[i] if i < len(burns_f) else None)
                    })
                df_h = pd.DataFrame(rows)
            elif burns_t:
                df_h = pd.DataFrame([{'Hour': i + 1, 'Tamarack (lb)': v} for i, v in enumerate(burns_t)])
            else:
                df_h = pd.DataFrame([{'Hour': i + 1, 'Flatwing (lb)': v} for i, v in enumerate(burns_f)])

            # Hide index for hourly fuel burn summary
            try:
                st.dataframe(df_h, use_container_width=True, hide_index=True)
            except TypeError:
                try:
                    st.dataframe(df_h.style.hide(axis='index'), use_container_width=True)
                except Exception:
                    df_h = df_h.reset_index(drop=True)
                    st.dataframe(df_h, use_container_width=True)
        except Exception:
            st.table(pd.DataFrame(rows))

    if modes_summary_df is not None and isinstance(modes_summary_df, pd.DataFrame) and not modes_summary_df.empty:
        st.subheader("Cruise Mode Summary")
        try:
            df_display = modes_summary_df.copy()
            if 'Chosen' in df_display.columns:
                df_display = df_display.drop(columns=['Chosen'])
            # Make headers tighter by shortening labels (no newline wrapping)
            col_renames = {}
            if 'Flatwing Fuel Used (lb)' in df_display.columns:
                col_renames['Flatwing Fuel Used (lb)'] = 'Flatwing Fuel'
            if 'Tamarack Fuel Used (lb)' in df_display.columns:
                col_renames['Tamarack Fuel Used (lb)'] = 'Tamarack Fuel'
            if 'Fuel Saved (lb)' in df_display.columns:
                col_renames['Fuel Saved (lb)'] = 'Fuel Saved (lb)'
            if 'Fuel Saved (gal)' in df_display.columns:
                col_renames['Fuel Saved (gal)'] = 'Fuel Saved (gal)'
            if 'Flatwing Time (min)' in df_display.columns:
                col_renames['Flatwing Time (min)'] = 'Flatwing Time'
            if 'Tamarack Time (min)' in df_display.columns:
                col_renames['Tamarack Time (min)'] = 'Tamarack Time'
            if 'Fuel Used (lb)' in df_display.columns:
                col_renames['Fuel Used (lb)'] = 'Fuel Used (lb)'
            if 'Total Time (min)' in df_display.columns:
                col_renames['Total Time (min)'] = 'Total Time (min)'
            if col_renames:
                df_display = df_display.rename(columns=col_renames)
            st.dataframe(df_display, use_container_width=True, hide_index=True)
        except TypeError:
            df_display = modes_summary_df.copy()
            if 'Chosen' in df_display.columns:
                df_display = df_display.drop(columns=['Chosen'])
            col_renames = {}
            if 'Flatwing Fuel Used (lb)' in df_display.columns:
                col_renames['Flatwing Fuel Used (lb)'] = 'Flatwing Fuel'
            if 'Tamarack Fuel Used (lb)' in df_display.columns:
                col_renames['Tamarack Fuel Used (lb)'] = 'Tamarack Fuel'
            if 'Fuel Saved (lb)' in df_display.columns:
                col_renames['Fuel Saved (lb)'] = 'Fuel Saved (lb)'
            if 'Fuel Saved (gal)' in df_display.columns:
                col_renames['Fuel Saved (gal)'] = 'Fuel Saved (gal)'
            if 'Flatwing Time (min)' in df_display.columns:
                col_renames['Flatwing Time (min)'] = 'Flatwing Time'
            if 'Tamarack Time (min)' in df_display.columns:
                col_renames['Tamarack Time (min)'] = 'Tamarack Time'
            if 'Fuel Used (lb)' in df_display.columns:
                col_renames['Fuel Used (lb)'] = 'Fuel Used (lb)'
            if 'Total Time (min)' in df_display.columns:
                col_renames['Total Time (min)'] = 'Total Time (min)'
            if col_renames:
                df_display = df_display.rename(columns=col_renames)
            try:
                st.dataframe(df_display.style.hide(axis='index'), use_container_width=True)
            except Exception:
                df_display = df_display.reset_index(drop=True)
                st.dataframe(df_display, use_container_width=True)

    figs = plot_flight_profiles(tamarack_data, flatwing_data, tamarack_results, flatwing_results)
    # Add the route map to figs so it can be embedded in the PDF
    try:
        figs["route_map"] = fig
    except Exception:
        pass

    # PDF export section
    st.subheader("Report Export")
    if not REPORTLAB_AVAILABLE:
        st.info("To enable PDF export with charts, install dependencies: pip install -U reportlab kaleido")
    else:
        # Build PDF into memory and offer for download
        def build_pdf_bytes():
            buf = BytesIO()
            c = canvas.Canvas(buf, pagesize=letter)
            width, height = letter

            y = height - 40
            c.setFont("Helvetica-Bold", 16)
            c.drawString(40, y, "Recover Mission Simulation Report")
            y -= 18
            c.setFont("Helvetica", 10)
            c.drawString(40, y, f"Route: {departure_airport}  {arrival_airport} | Distance: {distance_nm:.1f} NM | Bearing: {bearing_deg:.1f} ")
            y -= 14
            c.drawString(40, y, f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
            y -= 16

            # Airports table (mirror UI): with Airport, ICAO, PA (ft), DA (ft)
            try:
                airports_df = load_airports()
                dep_row = airports_df[airports_df['ident'] == departure_airport].iloc[0]
                arr_row = airports_df[airports_df['ident'] == arrival_airport].iloc[0]

                def short_name(n: str) -> str:
                    s = str(n).title()
                    s = (s.replace("International", "Intl")
                           .replace("Municipal", "Muni")
                           .replace("Regional", "Rgnl")
                           .replace("County", "Cnty")
                           .replace("Airport", "Arpt"))
                    return (s[:28] + "…") if len(s) > 29 else s

                pa_dep = float(dep_row.get('elevation_ft', 0.0))
                pa_arr = float(arr_row.get('elevation_ft', 0.0))
                da_dep = pa_dep + 120.0 * float(isa_dev)
                da_arr = pa_arr + 120.0 * float(isa_dev)

                c.setFont("Helvetica-Bold", 12)
                c.drawString(40, y, "Airports")
                y -= 14
                # Header row
                c.setFont("Helvetica-Bold", 10)
                c.drawString(40, y, "Type")
                c.drawString(90, y, "Airport")
                c.drawString(260, y, "ICAO")
                c.drawString(310, y, "PA (ft)")
                c.drawString(370, y, "DA (ft)")
                y -= 12
                c.setFont("Helvetica", 10)
                # Departure row
                c.drawString(40, y, "Departure")
                c.drawString(90, y, short_name(dep_row.get('name', departure_airport)))
                c.drawString(260, y, str(departure_airport))
                c.drawString(310, y, f"{int(round(pa_dep)):,}")
                c.drawString(370, y, f"{int(round(da_dep)):,}")
                y -= 12
                # Arrival row
                c.drawString(40, y, "Arrival")
                c.drawString(90, y, short_name(arr_row.get('name', arrival_airport)))
                c.drawString(260, y, str(arrival_airport))
                c.drawString(310, y, f"{int(round(pa_arr)):,}")
                c.drawString(370, y, f"{int(round(da_arr)):,}")
                y -= 14
                # Distance/Bearing line
                c.setFont("Helvetica", 10)
                c.drawString(40, y, f"Route Distance: {distance_nm:.1f} NM  |  Bearing: {bearing_deg:.1f}°")
                y -= 16
            except Exception:
                pass

            # Helper: embed a plotly figure by key
            def draw_fig(key, label):
                nonlocal y
                if key in figs and figs[key] is not None:
                    try:
                        img_bytes = figs[key].to_image(format="png", scale=2)
                        img = ImageReader(BytesIO(img_bytes))
                        img_w = width - 80
                        img_h = img_w * 0.6
                        if y - img_h < 40:
                            c.showPage(); y = height - 40
                        c.setFont("Helvetica-Bold", 12)
                        c.drawString(40, y, label)
                        y -= 12
                        c.drawImage(img, 40, y - img_h, width=img_w, height=img_h)
                        y -= img_h + 12
                    except Exception as e:
                        c.setFont("Helvetica-Oblique", 10)
                        c.drawString(40, y, f"(Could not embed chart '{label}': {str(e)})")
                        y -= 12

            # Helper: draw a weight status table within the PDF
            def draw_weight_table(title: str, df: pd.DataFrame, x0: int, y0: int) -> int:
                yloc = y0
                if df is None or df.empty:
                    return yloc
                # Column positions: component left; numeric columns right-aligned
                col_comp = x0
                col_wt = x0 + 300
                col_max = x0 + 460
                c.setFont("Helvetica-Bold", 12)
                c.drawString(x0, yloc, title)
                yloc -= 14
                c.setFont("Helvetica-Bold", 10)
                c.drawString(col_comp, yloc, "Component")
                c.drawString(col_wt - 70, yloc, "Weight (lb)")
                c.drawString(col_max - 90, yloc, "Max Weight (lb)")
                yloc -= 12
                c.setFont("Helvetica", 10)
                for _, row in df.iterrows():
                    if yloc < 60:
                        c.showPage(); yloc = height - 40; c.setFont("Helvetica", 10)
                        c.setFont("Helvetica-Bold", 10)
                        c.drawString(col_comp, yloc, "Component")
                        c.drawString(col_wt - 70, yloc, "Weight (lb)")
                        c.drawString(col_max - 90, yloc, "Max Weight (lb)")
                        yloc -= 12; c.setFont("Helvetica", 10)
                    comp = str(row.get('Component', ''))
                    wt = str(row.get('Weight (lb)', ''))
                    mwt = str(row.get('Max Weight (lb)', ''))
                    c.drawString(col_comp, yloc, comp)
                    if wt:
                        c.drawRightString(col_wt, yloc, wt)
                    if mwt:
                        c.drawRightString(col_max, yloc, mwt)
                    yloc -= 12
                return yloc

            # Helper: write metrics in a column starting at (x0, y0); returns new y
            def write_metrics_block_column(title, results, x0, y0):
                yloc = y0
                if yloc < 120:
                    c.showPage(); yloc = height - 40
                c.setFont("Helvetica-Bold", 12)
                c.drawString(x0, yloc, title)
                yloc -= 14
                c.setFont("Helvetica-Bold", 11)
                c.drawString(x0 + 10, yloc, "Takeoff")
                yloc -= 12
                c.setFont("Helvetica", 10)
                for k in [
                    "Takeoff Start Weight (lb)", "Takeoff End Weight (lb)",
                    "Takeoff Roll Dist (ft)", "Dist to 35 ft (ft)",
                    "Segment 1 Gradient (%)", "Dist to 400 ft (ft)", "Segment 2 Gradient (%)",
                    "Dist to 1500 ft (ft)", "Segment 3 Gradient (%)",
                    "Fuel Remaining After Takeoff (lb)"
                ]:
                    if k in results and results[k] is not None:
                        c.drawString(x0 + 20, yloc, f"{k}: {results[k]}")
                        yloc -= 12
                tvs = results.get("Takeoff V-Speeds")
                if isinstance(tvs, dict):
                    c.drawString(x0 + 20, yloc, "Takeoff V-Speeds:")
                    yloc -= 12
                    for vk in ["Weight", "VR", "V1", "V2", "V3"]:
                        if vk in tvs and tvs[vk] is not None:
                            c.drawString(x0 + 30, yloc, f"{vk}: {tvs[vk]}")
                            yloc -= 12
                # Climb
                c.setFont("Helvetica-Bold", 11)
                c.drawString(x0 + 10, yloc, "Climb")
                yloc -= 12
                c.setFont("Helvetica", 10)
                for k in [
                    "Climb Start Weight (lb)", "Climb End Weight (lb)",
                    "Climb Time (min)", "Climb Dist (NM)", "Climb Fuel (lb)", "Fuel Remaining After Climb (lb)"
                ]:
                    if k in results and results[k] is not None:
                        c.drawString(x0 + 20, yloc, f"{k}: {results[k]}")
                        yloc -= 12
                # Cruise
                c.setFont("Helvetica-Bold", 11)
                c.drawString(x0 + 10, yloc, "Cruise")
                yloc -= 12
                c.setFont("Helvetica", 10)
                for k in [
                    "Cruise Start Weight (lb)", "Cruise End Weight (lb)",
                    "Cruise Time (min)", "Cruise Dist (NM)", "Cruise Fuel (lb)",
                    "Cruise VKTAS (knots)", "Cruise Max Mach", "Cruise - First Level-Off Alt (ft)",
                    "Fuel Remaining After Cruise (lb)"
                ]:
                    if k in results and results[k] is not None:
                        c.drawString(x0 + 20, yloc, f"{k}: {results[k]}")
                        yloc -= 12
                # Descent
                c.setFont("Helvetica-Bold", 11)
                c.drawString(x0 + 10, yloc, "Descent")
                yloc -= 12
                c.setFont("Helvetica", 10)
                for k in [
                    "Descent Start Weight (lb)", "Descent End Weight (lb)",
                    "Descent Time (min)", "Descent Dist (NM)", "Descent Fuel (lb)",
                    "Fuel Remaining After Descent (lb)"
                ]:
                    if k in results and results[k] is not None:
                        c.drawString(x0 + 20, yloc, f"{k}: {results[k]}")
                        yloc -= 12
                # Landing
                c.setFont("Helvetica-Bold", 11)
                c.drawString(x0 + 10, yloc, "Landing")
                yloc -= 12
                c.setFont("Helvetica", 10)
                for k in [
                    "Landing Start Weight (lb)", "Landing End Weight (lb)",
                    "Landing - Dist from 35 ft to Stop (ft)", "Landing - Ground Roll (ft)",
                    "Fuel Remaining After Landing (lb)"
                ]:
                    if k in results and results[k] is not None:
                        c.drawString(x0 + 20, yloc, f"{k}: {results[k]}")
                        yloc -= 12
                avs = results.get("Approach V-Speeds")
                if isinstance(avs, dict):
                    c.drawString(x0 + 20, yloc, "Approach V-Speeds:")
                    yloc -= 12
                    for vk in ["Weight", "VAPP", "VREF"]:
                        if vk in avs and avs[vk] is not None:
                            c.drawString(x0 + 30, yloc, f"{vk}: {avs[vk]}")
                            yloc -= 12
                # Totals
                c.setFont("Helvetica-Bold", 11)
                c.drawString(x0 + 10, yloc, "Totals")
                yloc -= 12
                c.setFont("Helvetica", 10)
                for k in [
                    "Total Time (min)", "Total Dist (NM)", "Total Fuel Burned (lb)", "Fuel Remaining (lb)", "V1 Cut"
                ]:
                    if k in results and results[k] is not None:
                        c.drawString(x0 + 20, yloc, f"{k}: {results[k]}")
                        yloc -= 12
                return yloc

            def write_metrics_block(title, results):
                nonlocal y
                if y < 120:
                    c.showPage(); y = height - 40
                c.setFont("Helvetica-Bold", 12)
                c.drawString(40, y, title)
                y -= 14
                c.setFont("Helvetica", 10)
                # Takeoff section
                c.setFont("Helvetica-Bold", 11)
                c.drawString(50, y, "Takeoff")
                y -= 12
                c.setFont("Helvetica", 10)
                for k in [
                    "Takeoff Start Weight (lb)", "Takeoff End Weight (lb)",
                    "Takeoff Roll Dist (ft)", "Dist to 35 ft (ft)",
                    "Segment 1 Gradient (%)", "Dist to 400 ft (ft)", "Segment 2 Gradient (%)",
                    "Dist to 1500 ft (ft)", "Segment 3 Gradient (%)",
                    "Fuel Remaining After Takeoff (lb)"
                ]:
                    if y < 60:
                        c.showPage(); y = height - 40; c.setFont("Helvetica", 10)
                    if k in results and results[k] is not None:
                        c.drawString(60, y, f"{k}: {results[k]}")
                        y -= 12
                # Takeoff V-speeds if available
                tvs = results.get("Takeoff V-Speeds")
                if isinstance(tvs, dict):
                    c.drawString(60, y, "Takeoff V-Speeds:")
                    y -= 12
                    for vk in ["Weight", "VR", "V1", "V2", "V3"]:
                        if y < 60:
                            c.showPage(); y = height - 40; c.setFont("Helvetica", 10)
                        if vk in tvs and tvs[vk] is not None:
                            c.drawString(70, y, f"{vk}: {tvs[vk]}")
                            y -= 12
                if y < 120:
                    c.showPage(); y = height - 40
                # Climb section
                c.setFont("Helvetica-Bold", 11)
                c.drawString(50, y, "Climb")
                y -= 12
                c.setFont("Helvetica", 10)
                for k in ["Climb Time (min)", "Climb Dist (NM)", "Climb Fuel (lb)", "Fuel Remaining After Climb (lb)"]:
                    if y < 60:
                        c.showPage(); y = height - 40; c.setFont("Helvetica", 10)
                    if k in results and results[k] is not None:
                        c.drawString(60, y, f"{k}: {results[k]}")
                        y -= 12
                # Cruise section
                c.setFont("Helvetica-Bold", 11)
                c.drawString(50, y, "Cruise")
                y -= 12
                c.setFont("Helvetica", 10)
                for k in [
                    "Cruise Time (min)", "Cruise Dist (NM)", "Cruise Fuel (lb)",
                    "Cruise VKTAS (knots)", "Cruise Max Mach", "Cruise - First Level-Off Alt (ft)",
                    "Fuel Remaining After Cruise (lb)"
                ]:
                    if y < 60:
                        c.showPage(); y = height - 40; c.setFont("Helvetica", 10)
                    if k in results and results[k] is not None:
                        c.drawString(60, y, f"{k}: {results[k]}")
                        y -= 12
                if y < 120:
                    c.showPage(); y = height - 40
                # Descent section
                c.setFont("Helvetica-Bold", 11)
                c.drawString(50, y, "Descent")
                y -= 12
                c.setFont("Helvetica", 10)
                for k in [
                    "Descent Time (min)", "Descent Dist (NM)", "Descent Fuel (lb)",
                    "Fuel Remaining After Descent (lb)"
                ]:
                    if y < 60:
                        c.showPage(); y = height - 40; c.setFont("Helvetica", 10)
                    if k in results and results[k] is not None:
                        c.drawString(60, y, f"{k}: {results[k]}")
                        y -= 12
                # Landing section
                c.setFont("Helvetica-Bold", 11)
                c.drawString(50, y, "Landing")
                y -= 12
                c.setFont("Helvetica", 10)
                for k in [
                    "Landing Start Weight (lb)", "Landing End Weight (lb)",
                    "Landing - Dist from 35 ft to Stop (ft)", "Landing - Ground Roll (ft)",
                    "Fuel Remaining After Landing (lb)"
                ]:
                    if y < 60:
                        c.showPage(); y = height - 40; c.setFont("Helvetica", 10)
                    if k in results and results[k] is not None:
                        c.drawString(60, y, f"{k}: {results[k]}")
                        y -= 12
                # Approach V-speeds if available
                avs = results.get("Approach V-Speeds")
                if isinstance(avs, dict):
                    c.drawString(60, y, "Approach V-Speeds:")
                    y -= 12
                    for vk in ["Weight", "VAPP", "VREF"]:
                        if y < 60:
                            c.showPage(); y = height - 40; c.setFont("Helvetica", 10)
                        if vk in avs and avs[vk] is not None:
                            c.drawString(70, y, f"{vk}: {avs[vk]}")
                            y -= 12
                if y < 120:
                    c.showPage(); y = height - 40
                # Totals section
                c.setFont("Helvetica-Bold", 11)
                c.drawString(50, y, "Totals")
                y -= 12
                c.setFont("Helvetica", 10)
                for k in [
                    "Total Time (min)", "Total Dist (NM)", "Total Fuel Burned (lb)", "Fuel Remaining (lb)", "V1 Cut"
                ]:
                    if y < 60:
                        c.showPage(); y = height - 40; c.setFont("Helvetica", 10)
                    if k in results and results[k] is not None:
                        c.drawString(60, y, f"{k}: {results[k]}")
                        y -= 12

            # Flight Route Map at top (mirror UI)
            draw_fig("route_map", "Flight Route Map")

            # Weight Status tables (mirror UI)
            has_both = (weight_df_flatwing is not None and not weight_df_flatwing.empty) and (weight_df_tamarack is not None and not weight_df_tamarack.empty)
            has_single = (weight_df_single is not None and not weight_df_single.empty)
            if has_both or has_single:
                c.setFont("Helvetica-Bold", 12)
                c.drawString(40, y, "Weight Status")
                y -= 14
                if has_both:
                    # Render sequentially, full-width to avoid sideways overlap
                    y = draw_weight_table("Flatwing", weight_df_flatwing, 40, y)
                    if y < 120:
                        c.showPage(); y = height - 40
                    y = draw_weight_table("Tamarack", weight_df_tamarack, 40, y)
                elif has_single:
                    y = draw_weight_table("Weight Status", weight_df_single, 40, y)
                if y < 120:
                    c.showPage(); y = height - 40

            # Sequential metrics blocks to avoid truncation across pages
            write_metrics_block("Tamarack", tamarack_results)
            write_metrics_block("Flatwing", flatwing_results)
            if y < 120:
                c.showPage(); y = height - 40

            # Hourly Fuel Burn table (mirror UI)
            def hourly_burns(df: pd.DataFrame) -> list[int]:
                try:
                    if df is None or df.empty:
                        return []
                    if 'Time (hr)' not in df.columns or 'Fuel Remaining (lb)' not in df.columns:
                        return []
                    t = pd.to_numeric(df['Time (hr)'], errors='coerce')
                    f = pd.to_numeric(df['Fuel Remaining (lb)'], errors='coerce')
                    mask = t.notna() & f.notna()
                    t = t[mask].to_numpy()
                    f = f[mask].to_numpy()
                    if t.size == 0:
                        return []
                    t_max = float(np.nanmax(t))
                    if t_max <= 0:
                        return []
                    hours = int(np.ceil(t_max))
                    order = np.argsort(t)
                    t = t[order]; f = f[order]
                    burns = []
                    for h in range(hours):
                        start_t = float(h); end_t = float(min(h + 1, t_max))
                        if end_t <= start_t:
                            continue
                        start_f = float(np.interp(start_t, t, f))
                        end_f = float(np.interp(end_t, t, f))
                        burns.append(int(max(0.0, start_f - end_f)))
                    return burns
                except Exception:
                    return []

            burns_t = hourly_burns(tamarack_data)
            burns_f = hourly_burns(flatwing_data)
            c.setFont("Helvetica-Bold", 12)
            c.drawString(40, y, "Hourly Fuel Burn (lb)")
            y -= 16
            c.setFont("Helvetica-Bold", 10)
            c.drawString(50, y, "Hour")
            c.drawString(110, y, "Tamarack (lb)")
            c.drawString(230, y, "Flatwing (lb)")
            y -= 12
            c.setFont("Helvetica", 10)
            n = max(len(burns_t), len(burns_f))
            for i in range(n):
                if y < 60:
                    c.showPage(); y = height - 40
                    c.setFont("Helvetica-Bold", 12)
                    c.drawString(40, y, "Hourly Fuel Burn (lb) (cont)")
                    y -= 16
                    c.setFont("Helvetica-Bold", 10)
                    c.drawString(50, y, "Hour"); c.drawString(110, y, "Tamarack (lb)"); c.drawString(230, y, "Flatwing (lb)")
                    y -= 12; c.setFont("Helvetica", 10)
                c.drawString(50, y, str(i+1))
                if i < len(burns_t): c.drawString(110, y, str(burns_t[i]))
                if i < len(burns_f): c.drawString(230, y, str(burns_f[i]))
                y -= 12

            # Plots at the end (like UI order)
            if y < 120:
                c.showPage(); y = height - 40
            draw_fig("fuel_remaining", "Fuel Remaining vs. Distance")
            draw_fig("alt_mach", "Altitude and Mach vs. Distance")
            draw_fig("alt_tas_ias", "Altitude, TAS and IAS vs. Distance")
            draw_fig("ROC (fpm)", "Rate of Climb vs. Distance")
            draw_fig("Thrust (lb)", "Thrust vs. Distance")
            draw_fig("Drag (lb)", "Drag vs. Distance")

            c.save()
            buf.seek(0)
            return buf.getvalue()

        try:
            pdf_bytes = build_pdf_bytes()
            st.download_button(
                label="Download PDF Report",
                data=pdf_bytes,
                file_name="mission_report.pdf",
                mime="application/pdf"
            )
            # Also save to output folder with timestamped name
            try:
                if report_output_dir:
                    os.makedirs(report_output_dir, exist_ok=True)
                    pdf_name = f"Report_{departure_airport}_{arrival_airport}_{get_global_timestamp()}.pdf"
                    save_path = os.path.join(report_output_dir, pdf_name)
                    with open(save_path, "wb") as f:
                        f.write(pdf_bytes)
                    st.success(f"Saved PDF to: {save_path}")
            except Exception as e:
                st.warning(f"Could not save PDF to output folder: {e}")
        except Exception as e:
            st.warning(f"PDF export unavailable: {e}")

    