import streamlit as st
import plotly.graph_objects as go
import plotly.io as pio
from io import BytesIO
import datetime
import os
import numpy as np
import pandas as pd
from simulation import get_global_timestamp
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
    winds_temps_source, cruise_alt, departure_airport, arrival_airport, initial_fuel,
    report_output_dir=None
):
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
        if burns_t and burns_f:
            n = max(len(burns_t), len(burns_f))
            rows = []
            for i in range(n):
                rows.append({
                    'Hour': i + 1,
                    'Tamarack (lb)': (burns_t[i] if i < len(burns_t) else None),
                    'Flatwing (lb)': (burns_f[i] if i < len(burns_f) else None)
                })
            st.table(pd.DataFrame(rows))
        elif burns_t:
            rows = [{'Hour': i + 1, 'Tamarack (lb)': v} for i, v in enumerate(burns_t)]
            st.table(pd.DataFrame(rows))
        else:
            rows = [{'Hour': i + 1, 'Flatwing (lb)': v} for i, v in enumerate(burns_f)]
            st.table(pd.DataFrame(rows))

    figs = plot_flight_profiles(tamarack_data, flatwing_data, tamarack_results, flatwing_results)

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

            # Try to embed key charts
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

            # First page: Altitude/Mach
            draw_fig("alt_mach", "Altitude and Mach vs. Distance")
            c.showPage(); y = height - 40
            # Second page: Fuel Remaining
            draw_fig("fuel_remaining", "Fuel Remaining vs. Distance")
            # Additional pages: Performance charts
            draw_fig("ROC (fpm)", "Rate of Climb vs. Distance")
            draw_fig("Thrust (lb)", "Thrust vs. Distance")
            draw_fig("Drag (lb)", "Drag vs. Distance")

            def write_metrics_block(title, results):
                nonlocal y
                if y < 120:
                    c.showPage(); y = height - 40
                c.setFont("Helvetica-Bold", 12)
                c.drawString(40, y, title)
                y -= 14
                c.setFont("Helvetica", 10)
                for k in [
                    "Total Time (min)", "Total Dist (NM)", "Total Fuel Burned (lb)",
                    "Cruise VKTAS (knots)", "Cruise Max Mach"
                ]:
                    if k in results:
                        c.drawString(50, y, f"{k}: {results[k]}")
                        y -= 12

            write_metrics_block("Tamarack Summary", tamarack_results)
            write_metrics_block("Flatwing Summary", flatwing_results)

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

    