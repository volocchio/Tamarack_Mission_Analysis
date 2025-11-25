import streamlit as st
import plotly.graph_objects as go


def display_simulation_results(
    tamarack_data, tamarack_results, flatwing_data, flatwing_results, v1_cut_enabled,
    dep_latitude, dep_longitude, arr_latitude, arr_longitude, distance_nm, bearing_deg, winds_temps_source, cruise_alt,
    departure_airport, arrival_airport, initial_fuel
):
    """
    Display the simulation results, including V-speeds, final metrics, and charts for one or both models.

    Args:
        tamarack_data, tamarack_results: Simulation data and results for the Tamarack model.
        flatwing_data, flatwing_results: Simulation data and results for the Flatwing model.
        v1_cut_enabled: Whether V1 cut was enabled in the simulation.
        dep_latitude, dep_longitude: Coordinates of departure airport.
        arr_latitude, arr_longitude: Coordinates of arrival airport.
        distance_nm: Distance between airports in nautical miles.
        bearing_deg: Bearing from departure to arrival in degrees.
        winds_temps_source: Source of winds and temperatures data.
        cruise_alt: Cruise altitude in feet.
        departure_airport: Departure airport ICAO code.
        arrival_airport: Arrival airport ICAO code.
        initial_fuel: Initial fuel load in pounds.
    """

    # Create the map figure
    fig = go.Figure()

    # Add route line
    fig.add_trace(
        go.Scattergeo(
            lat=[dep_latitude, arr_latitude],
            lon=[dep_longitude, arr_longitude],
            mode='lines',
            line=dict(width=2, color='blue'),
            name='Route'
        )
    )

    # Add departure and arrival airports
    fig.add_trace(
        go.Scattergeo(
            lat=[dep_latitude],
            lon=[dep_longitude],
            mode='markers',
            marker=dict(size=10, color='green'),
            name='Departure'
        )
    )

    fig.add_trace(
        go.Scattergeo(
            lat=[arr_latitude],
            lon=[arr_longitude],
            mode='markers',
            marker=dict(size=10, color='red'),
            name='Arrival'
        )
    )

    # Configure map layout with proper zoom padding
    fig.update_layout(
        geo=dict(
            scope='north america',  
            showland=True,
            landcolor='rgb(243, 243, 243)',
            countrycolor='rgb(204, 204, 204)',
            showcountries=True,
            projection_type='equirectangular',
            center=dict(
                lat=(dep_latitude + arr_latitude) / 2,
                lon=(dep_longitude + arr_longitude) / 2
            ),
            lataxis=dict(
                range=[
                    min(dep_latitude, arr_latitude) - 5,  
                    max(dep_latitude, arr_latitude) + 5
                ]
            ),
            lonaxis=dict(
                range=[
                    min(dep_longitude, arr_longitude) - 10,  
                    max(dep_longitude, arr_longitude) + 10
                ]
            )
        ),
        title='Flight Route'
    )

    st.plotly_chart(fig, use_container_width=True)
    # Display distance and bearing
    st.success(f"Distance: {distance_nm:.1f} NM | Bearing: {bearing_deg:.1f}°")

    # Create two columns for side-by-side display
    col1, col2 = st.columns(2)
    
    # Tamarack column
    with col1:
        st.subheader("Tamarack")
        
        # Takeoff Section
        if tamarack_results:
            st.write("**Takeoff**")
            if "Takeoff V-Speeds" in tamarack_results:
                v_speeds = tamarack_results["Takeoff V-Speeds"]
                if v_speeds:
                    st.write("*V-Speeds*")
                    st.write(f"Weight: {v_speeds.get('Weight', 'N/A')} lb")
                    st.write(f"VR: {v_speeds.get('VR', 'N/A')} kts")
                    st.write(f"V1: {v_speeds.get('V1', 'N/A')} kts")
                    st.write(f"V2: {v_speeds.get('V2', 'N/A')} kts")
                    st.write(f"V3: {v_speeds.get('V3', 'N/A')} kts")
            st.write("*Performance*")
            st.write(f"Start Weight: {tamarack_results.get('Takeoff Start Weight (lb)', 'N/A')} lb")
            st.write(f"End Weight: {tamarack_results.get('Takeoff End Weight (lb)', 'N/A')} lb")
            st.write(f"Roll Distance: {tamarack_results.get('Takeoff Roll Dist (ft)', 'N/A')} ft")
            st.write(f"Dist to 35 ft: {tamarack_results.get('Dist to 35 ft (ft)', 'N/A')} ft")
            st.write(f"Seg 1 Grad: {tamarack_results.get('Segment 1 Gradient (%)', 'N/A')} %")
            st.write(f"Dist to 400 ft: {tamarack_results.get('Dist to 400 ft (ft)', 'N/A')} ft")
            st.write(f"Seg 2 Grad: {tamarack_results.get('Segment 2 Gradient (%)', 'N/A')} %")
            st.write(f"Dist to 1500 ft: {tamarack_results.get('Dist to 1500 ft (ft)', 'N/A')} ft")
            st.write(f"Seg 3 Grad: {tamarack_results.get('Segment 3 Gradient (%)', 'N/A')} %")
            st.write("---")
            
            # Add V1 cut information
            if tamarack_results.get("V1 Cut", False):
                st.write("**V1 Cut**")
                st.write("*Single Engine Operation*")
                st.write("---")
            else:
                # Climb Section
                st.write("**Climb**")
                st.write(f"Start Weight: {tamarack_results.get('Climb Start Weight (lb)', 'N/A')} lb")
                st.write(f"End Weight: {tamarack_results.get('Climb End Weight (lb)', 'N/A')} lb")
                st.write(f"Time: {tamarack_results.get('Climb Time (min)', 'N/A')} min")
                st.write(f"Distance: {tamarack_results.get('Climb Dist (NM)', 'N/A')} NM")
                st.write(f"Fuel: {tamarack_results.get('Climb Fuel (lb)', 'N/A')} lb")
                st.write("---")
                
                # Cruise Section
                st.write("**Cruise**")
                st.write(f"Start Weight: {tamarack_results.get('Cruise Start Weight (lb)', 'N/A')} lb")
                st.write(f"End Weight: {tamarack_results.get('Cruise End Weight (lb)', 'N/A')} lb")
                st.write(f"Time: {tamarack_results.get('Cruise Time (min)', 'N/A')} min")
                st.write(f"Distance: {tamarack_results.get('Cruise Dist (NM)', 'N/A')} NM")
                st.write(f"Fuel: {tamarack_results.get('Cruise Fuel (lb)', 'N/A')} lb")
                st.write(f"VKTAS: {tamarack_results.get('Cruise VKTAS (knots)', 'N/A')} kts")
                
                cruise_time = tamarack_results.get('Cruise Time (min)', None)
                cruise_fuel = tamarack_results.get('Cruise Fuel (lb)', None)
                if cruise_time is not None and cruise_time > 0:
                    fuel_rate = cruise_fuel / (cruise_time / 60)
                    st.write(f"Fuel Rate: {fuel_rate:.1f} lb/hr")
                st.write("---")
                
                # Descent Section
                st.write("**Descent**")
                st.write(f"Start Weight: {tamarack_results.get('Descent Start Weight (lb)', 'N/A')} lb")
                st.write(f"End Weight: {tamarack_results.get('Descent End Weight (lb)', 'N/A')} lb")
                st.write(f"Time: {tamarack_results.get('Descent Time (min)', 'N/A')} min")
                st.write(f"Distance: {tamarack_results.get('Descent Dist (NM)', 'N/A')} NM")
                st.write(f"Fuel: {tamarack_results.get('Descent Fuel (lb)', 'N/A')} lb")
                st.write("---")
                
                # Landing Section
                st.write("**Landing**")
                landing_start_weight = tamarack_results.get('Landing Start Weight (lb)', None)
                if landing_start_weight is not None:
                    st.write(f"Start Weight: {landing_start_weight} lb")
                
                dist_land_35 = tamarack_results.get('Landing - Dist from 35 ft to Stop (ft)', None)
                ground_roll = tamarack_results.get('Landing - Ground Roll (ft)', None)
                
                if dist_land_35 is not None:
                    st.write(f"Total Distance: {dist_land_35} ft")
                if ground_roll is not None:
                    st.write(f"Ground Roll: {ground_roll} ft")
                
                descent_fuel_burned = tamarack_results.get('Descent Fuel (lb)', None)
                if descent_fuel_burned is not None and landing_start_weight is not None:
                    final_weight = landing_start_weight - descent_fuel_burned
                    st.write(f"Final Weight: {int(final_weight)} lb")
                st.write("---")
                
                # Total Flight Section
                st.write("**Total Flight**")
                st.write(f"Time: {tamarack_results.get('Total Time (min)', 'N/A')} min")
                st.write(f"Distance: {tamarack_results.get('Total Dist (NM)', 'N/A')} NM")
                first_level_off_alt = tamarack_results.get('First Level-Off Alt (ft)', None)
                if first_level_off_alt is not None:
                    st.write(f"First Level-Off Alt: {first_level_off_alt:.0f} ft")
                else:
                    st.write("First Level-Off Alt: N/A ft")
                
                fuel_burned = tamarack_results.get('Total Fuel Burned (lb)', None)
                fuel_start = tamarack_results.get('Fuel Start (lb)', initial_fuel)  # Use initial_fuel as default
                fuel_end = fuel_start - fuel_burned if fuel_burned is not None else None
                
                if fuel_burned is not None:
                    st.write(f"Fuel Burned: {fuel_burned:.0f} lb")
                else:
                    st.write("Fuel Burned: N/A lb")
                
                if fuel_start is not None:
                    st.write(f"Fuel Start: {fuel_start:.0f} lb")
                else:
                    st.write("Fuel Start: N/A lb")
                
                if fuel_end is not None:
                    st.write(f"Fuel End: {fuel_end:.0f} lb")
                else:
                    st.write("Fuel End: N/A lb")
                st.write("---")

    # Flatwing column
    with col2:
        st.subheader("Flatwing")
        
        # Takeoff Section
        if flatwing_results:
            st.write("**Takeoff**")
            if "Takeoff V-Speeds" in flatwing_results:
                v_speeds = flatwing_results["Takeoff V-Speeds"]
                if v_speeds:
                    st.write("*V-Speeds*")
                    st.write(f"Weight: {v_speeds.get('Weight', 'N/A')} lb")
                    st.write(f"VR: {v_speeds.get('VR', 'N/A')} kts")
                    st.write(f"V1: {v_speeds.get('V1', 'N/A')} kts")
                    st.write(f"V2: {v_speeds.get('V2', 'N/A')} kts")
                    st.write(f"V3: {v_speeds.get('V3', 'N/A')} kts")
            st.write("*Performance*")
            st.write(f"Start Weight: {flatwing_results.get('Takeoff Start Weight (lb)', 'N/A')} lb")
            st.write(f"End Weight: {flatwing_results.get('Takeoff End Weight (lb)', 'N/A')} lb")
            st.write(f"Roll Distance: {flatwing_results.get('Takeoff Roll Dist (ft)', 'N/A')} ft")
            st.write(f"Dist to 35 ft: {flatwing_results.get('Dist to 35 ft (ft)', 'N/A')} ft")
            st.write(f"Seg 1 Grad: {flatwing_results.get('Segment 1 Gradient (%)', 'N/A')} %")
            st.write(f"Dist to 400 ft: {flatwing_results.get('Dist to 400 ft (ft)', 'N/A')} ft")
            st.write(f"Seg 2 Grad: {flatwing_results.get('Segment 2 Gradient (%)', 'N/A')} %")
            st.write(f"Dist to 1500 ft: {flatwing_results.get('Dist to 1500 ft (ft)', 'N/A')} ft")
            st.write(f"Seg 3 Grad: {flatwing_results.get('Segment 3 Gradient (%)', 'N/A')} %")
            st.write("---")
            
            # Add V1 cut information
            if flatwing_results.get("V1 Cut", False):
                st.write("**V1 Cut**")
                st.write("*Single Engine Operation*")
                st.write("---")
            else:
                # Climb Section
                st.write("**Climb**")
                st.write(f"Start Weight: {flatwing_results.get('Climb Start Weight (lb)', 'N/A')} lb")
                st.write(f"End Weight: {flatwing_results.get('Climb End Weight (lb)', 'N/A')} lb")
                st.write(f"Time: {flatwing_results.get('Climb Time (min)', 'N/A')} min")
                st.write(f"Distance: {flatwing_results.get('Climb Dist (NM)', 'N/A')} NM")
                st.write(f"Fuel: {flatwing_results.get('Climb Fuel (lb)', 'N/A')} lb")
                st.write("---")
                
                # Cruise Section
                st.write("**Cruise**")
                st.write(f"Start Weight: {flatwing_results.get('Cruise Start Weight (lb)', 'N/A')} lb")
                st.write(f"End Weight: {flatwing_results.get('Cruise End Weight (lb)', 'N/A')} lb")
                st.write(f"Time: {flatwing_results.get('Cruise Time (min)', 'N/A')} min")
                st.write(f"Distance: {flatwing_results.get('Cruise Dist (NM)', 'N/A')} NM")
                st.write(f"Fuel: {flatwing_results.get('Cruise Fuel (lb)', 'N/A')} lb")
                st.write(f"VKTAS: {flatwing_results.get('Cruise VKTAS (knots)', 'N/A')} kts")
                
                cruise_time = flatwing_results.get('Cruise Time (min)', None)
                cruise_fuel = flatwing_results.get('Cruise Fuel (lb)', None)
                if cruise_time is not None and cruise_time > 0:
                    fuel_rate = cruise_fuel / (cruise_time / 60)
                    st.write(f"Fuel Rate: {fuel_rate:.1f} lb/hr")
                st.write("---")
                
                # Descent Section
                st.write("**Descent**")
                st.write(f"Start Weight: {flatwing_results.get('Descent Start Weight (lb)', 'N/A')} lb")
                st.write(f"End Weight: {flatwing_results.get('Descent End Weight (lb)', 'N/A')} lb")
                st.write(f"Time: {flatwing_results.get('Descent Time (min)', 'N/A')} min")
                st.write(f"Distance: {flatwing_results.get('Descent Dist (NM)', 'N/A')} NM")
                st.write(f"Fuel: {flatwing_results.get('Descent Fuel (lb)', 'N/A')} lb")
                st.write("---")
                
                # Landing Section
                st.write("**Landing**")
                landing_start_weight = flatwing_results.get('Landing Start Weight (lb)', None)
                if landing_start_weight is not None:
                    st.write(f"Start Weight: {landing_start_weight} lb")
                
                dist_land_35 = flatwing_results.get('Landing - Dist from 35 ft to Stop (ft)', None)
                ground_roll = flatwing_results.get('Landing - Ground Roll (ft)', None)
                
                if dist_land_35 is not None:
                    st.write(f"Total Distance: {dist_land_35} ft")
                if ground_roll is not None:
                    st.write(f"Ground Roll: {ground_roll} ft")
                
                descent_fuel_burned = flatwing_results.get('Descent Fuel (lb)', None)
                if descent_fuel_burned is not None and landing_start_weight is not None:
                    final_weight = landing_start_weight - descent_fuel_burned
                    st.write(f"Final Weight: {int(final_weight)} lb")
                st.write("---")
                
                # Total Flight Section
                st.write("**Total Flight**")
                st.write(f"Time: {flatwing_results.get('Total Time (min)', 'N/A')} min")
                st.write(f"Distance: {flatwing_results.get('Total Dist (NM)', 'N/A')} NM")
                first_level_off_alt = flatwing_results.get('First Level-Off Alt (ft)', None)
                if first_level_off_alt is not None:
                    st.write(f"First Level-Off Alt: {first_level_off_alt:.0f} ft")
                else:
                    st.write("First Level-Off Alt: N/A ft")
                
                fuel_burned = flatwing_results.get('Total Fuel Burned (lb)', None)
                fuel_start = flatwing_results.get('Fuel Start (lb)', initial_fuel)  # Use initial_fuel as default
                fuel_end = fuel_start - fuel_burned if fuel_burned is not None else None
                
                if fuel_burned is not None:
                    st.write(f"Fuel Burned: {fuel_burned:.0f} lb")
                else:
                    st.write("Fuel Burned: N/A lb")
                
                if fuel_start is not None:
                    st.write(f"Fuel Start: {fuel_start:.0f} lb")
                else:
                    st.write("Fuel Start: N/A lb")
                
                if fuel_end is not None:
                    st.write(f"Fuel End: {fuel_end:.0f} lb")
                else:
                    st.write("Fuel End: N/A lb")
                st.write("---")

    # Check for exceedances or errors
    if tamarack_results.get("exceedances"):
        for msg in tamarack_results["exceedances"]:
            st.error(msg)
    elif tamarack_results.get("error"):
        st.error(tamarack_results["error"])
    
    if flatwing_results.get("exceedances"):
        for msg in flatwing_results["exceedances"]:
            st.error(msg)
    elif flatwing_results.get("error"):
        st.error(flatwing_results["error"])

    # Display graphs comparing both models if both have data
    if not tamarack_data.empty and not flatwing_data.empty:
        st.subheader("Flight Profile Visualizations (Tamarack vs. Flatwing)")

        try:
            # Altitude and Mach vs. Distance with secondary Y-axis for Mach
            st.write("**Altitude and Mach vs. Distance**")
            fig_alt_mach = go.Figure()
            # Tamarack Altitude
            fig_alt_mach.add_trace(
                go.Scatter(
                    x=tamarack_data['Distance (NM)'],
                    y=tamarack_data['Altitude (ft)'],
                    name='Altitude (ft) - Tamarack',
                    line=dict(color='blue')
                )
            )
            # Flatwing Altitude
            fig_alt_mach.add_trace(
                go.Scatter(
                    x=flatwing_data['Distance (NM)'],
                    y=flatwing_data['Altitude (ft)'],
                    name='Altitude (ft) - Flatwing',
                    line=dict(color='purple')
                )
            )
            # Tamarack Mach
            fig_alt_mach.add_trace(
                go.Scatter(
                    x=tamarack_data['Distance (NM)'],
                    y=tamarack_data['Mach'],
                    name='Mach - Tamarack',
                    yaxis='y2',
                    line=dict(color='orange')
                )
            )
            # Flatwing Mach
            fig_alt_mach.add_trace(
                go.Scatter(
                    x=flatwing_data['Distance (NM)'],
                    y=flatwing_data['Mach'],
                    name='Mach - Flatwing',
                    yaxis='y2',
                    line=dict(color='red')
                )
            )
            fig_alt_mach.update_layout(
                xaxis=dict(
                    title='Distance (NM)',
                    showgrid=True,
                    gridcolor='rgba(200, 200, 200, 0.5)',
                    gridwidth=1
                ),
                yaxis=dict(title='Altitude (ft)'),
                yaxis2=dict(
                    title='Mach',
                    overlaying='y',
                    side='right'
                ),
                legend=dict(x=0.1, y=1.1, orientation='h')
            )
            st.plotly_chart(fig_alt_mach)

            # Speed (KTAS and KIAS) vs. Distance
            st.write("**Speed (KTAS and KIAS) vs. Distance**")
            fig_speed = go.Figure()
            # Tamarack VKTAS
            fig_speed.add_trace(
                go.Scatter(
                    x=tamarack_data['Distance (NM)'],
                    y=tamarack_data['VKTAS (kts)'],
                    name='VKTAS - Tamarack',
                    line=dict(color='blue')
                )
            )
            # Flatwing VKTAS
            fig_speed.add_trace(
                go.Scatter(
                    x=flatwing_data['Distance (NM)'],
                    y=flatwing_data['VKTAS (kts)'],
                    name='VKTAS - Flatwing',
                    line=dict(color='purple')
                )
            )
            # Tamarack VKIAS
            fig_speed.add_trace(
                go.Scatter(
                    x=tamarack_data['Distance (NM)'],
                    y=tamarack_data['VKIAS (kts)'],
                    name='VKIAS - Tamarack',
                    line=dict(color='blue', dash='dot')
                )
            )
            # Flatwing VKIAS
            fig_speed.add_trace(
                go.Scatter(
                    x=flatwing_data['Distance (NM)'],
                    y=flatwing_data['VKIAS (kts)'],
                    name='VKIAS - Flatwing',
                    line=dict(color='purple', dash='dot')
                )
            )
            fig_speed.update_layout(
                title='Speed vs. Distance',
                xaxis_title='Distance (NM)',
                yaxis_title='Speed (kts)',
                legend_title='Speed Type'
            )
            st.plotly_chart(fig_speed)

            # Add VKTAS vs. Time chart
            st.write("**Speed (KTAS) vs. Time**")
            fig_speed_time = go.Figure()
            # Tamarack VKTAS
            fig_speed_time.add_trace(
                go.Scatter(
                    x=tamarack_data['Time (hr)'],
                    y=tamarack_data['VKTAS (kts)'],
                    name='VKTAS - Tamarack',
                    line=dict(color='blue')
                )
            )
            # Flatwing VKTAS
            fig_speed_time.add_trace(
                go.Scatter(
                    x=flatwing_data['Time (hr)'],
                    y=flatwing_data['VKTAS (kts)'],
                    name='VKTAS - Flatwing',
                    line=dict(color='purple')
                )
            )
            fig_speed_time.update_layout(
                title='Speed vs. Time',
                xaxis_title='Time (hr)',
                yaxis_title='Speed (kts)',
                legend_title='Speed Type'
            )
            st.plotly_chart(fig_speed_time)

            # Rate of Climb (ROC) and Gradient vs. Distance
            st.write("**Rate of Climb (ROC) and Gradient vs. Distance**")
            fig_roc = go.Figure()
            # Tamarack ROC
            fig_roc.add_trace(
                go.Scatter(
                    x=tamarack_data['Distance (NM)'],
                    y=tamarack_data['ROC (fpm)'],
                    name='ROC (fpm) - Tamarack',
                    line=dict(color='blue')
                )
            )
            # Flatwing ROC
            fig_roc.add_trace(
                go.Scatter(
                    x=flatwing_data['Distance (NM)'],
                    y=flatwing_data['ROC (fpm)'],
                    name='ROC (fpm) - Flatwing',
                    line=dict(color='purple')
                )
            )
            # Tamarack Gradient
            fig_roc.add_trace(
                go.Scatter(
                    x=tamarack_data['Distance (NM)'],
                    y=tamarack_data['Gradient (%)'],
                    name='Gradient (%) - Tamarack',
                    yaxis='y2',
                    line=dict(color='orange')
                )
            )
            # Flatwing Gradient
            fig_roc.add_trace(
                go.Scatter(
                    x=flatwing_data['Distance (NM)'],
                    y=flatwing_data['Gradient (%)'],
                    name='Gradient (%) - Flatwing',
                    yaxis='y2',
                    line=dict(color='red')
                )
            )
            fig_roc.update_layout(
                xaxis=dict(
                    title='Distance (NM)',
                    showgrid=True,
                    gridcolor='rgba(200, 200, 200, 0.5)',
                    gridwidth=1
                ),
                yaxis=dict(title='ROC (fpm)'),
                yaxis2=dict(
                    title='Gradient (%)',
                    overlaying='y',
                    side='right'
                ),
                legend=dict(x=0.1, y=1.1, orientation='h')
            )
            st.plotly_chart(fig_roc)

        except ValueError as e:
            st.warning(f"Error rendering charts: {e}")
            st.write("Displaying raw data instead:")
            st.write("Tamarack Data:")
            st.dataframe(tamarack_data)
            st.write("Flatwing Data:")
            st.dataframe(flatwing_data)

    # Display graphs for Tamarack only if Flatwing data is empty
    elif not tamarack_data.empty:
        st.subheader("Flight Profile Visualizations - Tamarack")
        try:
            # Altitude and Mach vs. Distance with secondary Y-axis for Mach
            st.write("**Altitude and Mach vs. Distance**")
            fig_alt_mach = go.Figure()
            fig_alt_mach.add_trace(
                go.Scatter(
                    x=tamarack_data['Distance (NM)'],
                    y=tamarack_data['Altitude (ft)'],
                    name='Altitude (ft) - Tamarack',
                    line=dict(color='blue')
                )
            )
            fig_alt_mach.add_trace(
                go.Scatter(
                    x=tamarack_data['Distance (NM)'],
                    y=tamarack_data['Mach'],
                    name='Mach - Tamarack',
                    yaxis='y2',
                    line=dict(color='orange')
                )
            )
            fig_alt_mach.update_layout(
                xaxis=dict(
                    title='Distance (NM)',
                    showgrid=True,
                    gridcolor='rgba(200, 200, 200, 0.5)',
                    gridwidth=1
                ),
                yaxis=dict(title='Altitude (ft)'),
                yaxis2=dict(
                    title='Mach',
                    overlaying='y',
                    side='right'
                ),
                legend=dict(x=0.1, y=1.1, orientation='h')
            )
            st.plotly_chart(fig_alt_mach)

            # Speed (KTAS and KIAS) vs. Distance
            st.write("**Speed (KTAS and KIAS) vs. Distance**")
            fig_speed = go.Figure()
            # Tamarack VKTAS
            fig_speed.add_trace(
                go.Scatter(
                    x=tamarack_data['Distance (NM)'],
                    y=tamarack_data['VKTAS (kts)'],
                    name='VKTAS - Tamarack',
                    line=dict(color='blue')
                )
            )
            # Tamarack VKIAS
            fig_speed.add_trace(
                go.Scatter(
                    x=tamarack_data['Distance (NM)'],
                    y=tamarack_data['VKIAS (kts)'],
                    name='VKIAS - Tamarack',
                    line=dict(color='blue', dash='dot')
                )
            )
            fig_speed.update_layout(
                title='Speed vs. Distance',
                xaxis_title='Distance (NM)',
                yaxis_title='Speed (kts)',
                legend_title='Speed Type'
            )
            st.plotly_chart(fig_speed)

            # Add VKTAS vs. Time chart
            st.write("**Speed (KTAS) vs. Time**")
            fig_speed_time = go.Figure()
            # Tamarack VKTAS
            fig_speed_time.add_trace(
                go.Scatter(
                    x=tamarack_data['Time (hr)'],
                    y=tamarack_data['VKTAS (kts)'],
                    name='VKTAS - Tamarack',
                    line=dict(color='blue')
                )
            )
            fig_speed_time.update_layout(
                title='Speed vs. Time',
                xaxis_title='Time (hr)',
                yaxis_title='Speed (kts)',
                legend_title='Speed Type'
            )
            st.plotly_chart(fig_speed_time)

            # Rate of Climb (ROC) and Gradient vs. Distance
            st.write("**Rate of Climb (ROC) and Gradient vs. Distance**")
            fig_roc = go.Figure()
            fig_roc.add_trace(
                go.Scatter(
                    x=tamarack_data['Distance (NM)'],
                    y=tamarack_data['ROC (fpm)'],
                    name='ROC (fpm) - Tamarack',
                    line=dict(color='blue')
                )
            )
            fig_roc.add_trace(
                go.Scatter(
                    x=tamarack_data['Distance (NM)'],
                    y=tamarack_data['Gradient (%)'],
                    name='Gradient (%) - Tamarack',
                    yaxis='y2',
                    line=dict(color='orange')
                )
            )
            fig_roc.update_layout(
                xaxis=dict(
                    title='Distance (NM)',
                    showgrid=True,
                    gridcolor='rgba(200, 200, 200, 0.5)',
                    gridwidth=1
                ),
                yaxis=dict(title='ROC (fpm)'),
                yaxis2=dict(
                    title='Gradient (%)',
                    overlaying='y',
                    side='right'
                ),
                legend=dict(x=0.1, y=1.1, orientation='h')
            )
            st.plotly_chart(fig_roc)

        except ValueError as e:
            st.warning(f"Error rendering charts: {e}")
            st.write("Displaying raw data instead:")
            st.write("Tamarack Data:")
            st.dataframe(tamarack_data)

    # Display graphs for Flatwing only if Tamarack data is empty
    elif not flatwing_data.empty:
        st.subheader("Flight Profile Visualizations - Flatwing")
        try:
            # Altitude and Mach vs. Distance with secondary Y-axis for Mach
            st.write("**Altitude and Mach vs. Distance**")
            fig_alt_mach = go.Figure()
            fig_alt_mach.add_trace(
                go.Scatter(
                    x=flatwing_data['Distance (NM)'],
                    y=flatwing_data['Altitude (ft)'],
                    name='Altitude (ft) - Flatwing',
                    line=dict(color='purple')
                )
            )
            fig_alt_mach.add_trace(
                go.Scatter(
                    x=flatwing_data['Distance (NM)'],
                    y=flatwing_data['Mach'],
                    name='Mach - Flatwing',
                    yaxis='y2',
                    line=dict(color='red')
                )
            )
            fig_alt_mach.update_layout(
                xaxis=dict(
                    title='Distance (NM)',
                    showgrid=True,
                    gridcolor='rgba(200, 200, 200, 0.5)',
                    gridwidth=1
                ),
                yaxis=dict(title='Altitude (ft)'),
                yaxis2=dict(
                    title='Mach',
                    overlaying='y',
                    side='right'
                ),
                legend=dict(x=0.1, y=1.1, orientation='h')
            )
            st.plotly_chart(fig_alt_mach)

            # Speed (KTAS and KIAS) vs. Distance
            st.write("**Speed (KTAS and KIAS) vs. Distance**")
            fig_speed = go.Figure()
            # Flatwing VKTAS
            fig_speed.add_trace(
                go.Scatter(
                    x=flatwing_data['Distance (NM)'],
                    y=flatwing_data['VKTAS (kts)'],
                    name='VKTAS - Flatwing',
                    line=dict(color='purple')
                )
            )
            # Flatwing VKIAS
            fig_speed.add_trace(
                go.Scatter(
                    x=flatwing_data['Distance (NM)'],
                    y=flatwing_data['VKIAS (kts)'],
                    name='VKIAS - Flatwing',
                    line=dict(color='purple', dash='dot')
                )
            )
            fig_speed.update_layout(
                title='Speed vs. Distance',
                xaxis_title='Distance (NM)',
                yaxis_title='Speed (kts)',
                legend_title='Speed Type'
            )
            st.plotly_chart(fig_speed)

            # Add VKTAS vs. Time chart
            st.write("**Speed (KTAS) vs. Time**")
            fig_speed_time = go.Figure()
            # Flatwing VKTAS
            fig_speed_time.add_trace(
                go.Scatter(
                    x=flatwing_data['Time (hr)'],
                    y=flatwing_data['VKTAS (kts)'],
                    name='VKTAS - Flatwing',
                    line=dict(color='purple')
                )
            )
            fig_speed_time.update_layout(
                title='Speed vs. Time',
                xaxis_title='Time (hr)',
                yaxis_title='Speed (kts)',
                legend_title='Speed Type'
            )
            st.plotly_chart(fig_speed_time)

            # Rate of Climb (ROC) and Gradient vs. Distance
            st.write("**Rate of Climb (ROC) and Gradient vs. Distance**")
            fig_roc = go.Figure()
            fig_roc.add_trace(
                go.Scatter(
                    x=flatwing_data['Distance (NM)'],
                    y=flatwing_data['ROC (fpm)'],
                    name='ROC (fpm) - Flatwing',
                    line=dict(color='purple')
                )
            )
            fig_roc.add_trace(
                go.Scatter(
                    x=flatwing_data['Distance (NM)'],
                    y=flatwing_data['Gradient (%)'],
                    name='Gradient (%) - Flatwing',
                    yaxis='y2',
                    line=dict(color='red')
                )
            )
            fig_roc.update_layout(
                xaxis=dict(
                    title='Distance (NM)',
                    showgrid=True,
                    gridcolor='rgba(200, 200, 200, 0.5)',
                    gridwidth=1
                ),
                yaxis=dict(title='ROC (fpm)'),
                yaxis2=dict(
                    title='Gradient (%)',
                    overlaying='y',
                    side='right'
                ),
                legend=dict(x=0.1, y=1.1, orientation='h')
            )
            st.plotly_chart(fig_roc)

        except ValueError as e:
            st.warning(f"Error rendering charts: {e}")
            st.write("Displaying raw data instead:")
            st.write("Flatwing Data:")
            st.dataframe(flatwing_data)