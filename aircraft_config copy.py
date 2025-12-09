"""
Aircraft Configuration Module

This module defines the AIRCRAFT_CONFIG dictionary, which contains configuration parameters
for different aircraft models and modifications.
"""

# AIRCRAFT_CONFIG dictionary maps (aircraft, mod) tuples to configuration tuples.
# Each configuration tuple contains the following parameters in order:
# Index 0:  s           - Wing area (ft^2)
# Index 1:  b           - Wing span (ft)
# Index 2:  e           - Oswald efficiency factor
# Index 3:  h           - Winglet height (ft)
# Index 4:  sweep_25c   - Wing sweep at 25% chord (degrees)
# Index 5:  SFC         - Specific Fuel Consumption (lb/hr/lb)
# Index 6:  engines     - Number of engines
# Index 7:  thrust_mult - Thrust multiplier (thrust per engine / reference thrust)
# Index 8:  ceiling     - Service ceiling (ft)
# Index 9:  CL0         - Zero-lift coefficient
# Index 10: CLA         - Lift curve slope (1/rad)
# Index 11: cdo         - Zero-lift drag coefficient
# Index 12: dcdo_flap1  - Drag coefficient increment for takeoff flaps 15
# Index 13: dcdo_flap2  - Drag coefficient increment for takeoff flaps 30
# Index 14: dcdo_flap3  - Drag coefficient increment for ground flaps and spoilers 40
# Index 15: dcdo_gear   - Drag coefficient increment for landing gear
# Index 16: mu_to       - Rolling friction coefficient during takeoff
# Index 17: mu_lnd      - Rolling friction coefficient during landing
# Index 18: bow         - Basic Operating Weight (Empty Weight + pilot) (lb)
# Index 19: MZFW        - Maximum Zero Fuel Weight (lb)
# Index 20: MRW         - Maximum Ramp Weight (lb)
# Index 21: MTOW        - Maximum Takeoff Weight (lb)
# Index 22: max_fuel    - Maximum fuel capacity (lb)
# Index 23: taxi_fuel   - Taxi fuel allowance (lb)
# Index 24: reserve_fuel - Reserve fuel requirement (lb)
# Index 25: mmo         - Maximum Mach number
# Index 26: VMO         - Maximum Operating Speed (kts)
# Index 27: Clmax       - Maximum lift coefficient (clean)
# Index 28: Clmax_1     - Maximum lift coefficient (flaps 15)
# Index 29: Clmax_2     - Maximum lift coefficient (flaps 40)
# Index 30: M_climb     - Mach number for climb
# Index 31: v_climb     - Climb speed (kts)
# Index 32: roc_min     - Minimum rate of climb (ft/min)
# Index 33: M_descent   - Mach number for descent
# Index 34: v_descent   - Descent speed (kts)

"""
Aircraft Configuration Module

This module defines the AIRCRAFT_CONFIG dictionary containing realistic performance
and weight parameters for various Cessna CitationJet models (with/without Tamarack
Active Winglets) and Cessna Grand Caravan models.

All weights are in pounds (lb), speeds in knots (kts), distances in feet (ft),
fuel consumption in lb/hr/lb (jets) or lb/shp-hr (turboprops).
"""

AIRCRAFT_CONFIG = {
    # ====================== CITATION M2 / CJ / CJ1 / CJ1+ ======================
    ('CJ', 'Flatwing'): (240.0, 46.5, 0.75, 0.0, 0, 0.72, 2, 0.674, 41000.0, 0.2, 4.5, 0.028,
                         0.01, 0.015, 0.011, 0.017, 0.015, 0.25, 6800.0, 8400, 10500.0, 10400.0,
                         3440.0, 100, 600, 0.70, 263, 1.35, 1.54, 1.75, 0.53, 200, 300, 0.70, 260),

    ('CJ', 'Tamarack'): (250.0, 51.5, 0.8025, 3.0, 0, 0.72, 2, 0.674, 41000.0, 0.21, 4.5*1.05, 0.026244,
                         0.01, 0.015, 0.011, 0.017, 0.015, 0.25, 6879.0, 8800, 10500.0, 10400.0,
                         3440.0, 100, 600, 0.70, 263, 1.4175, 1.617, 1.8375, 0.51, 180, 300, 0.70, 260),

    ('CJ1', 'Flatwing'): (240.0, 46.5, 0.75, 0.0, 0, 0.72, 2, 0.674, 41000.0, 0.2, 4.5, 0.028,
                          0.01, 0.015, 0.011, 0.017, 0.015, 0.25, 6900.0, 8400, 10700.0, 10600.0,
                          3440.0, 100, 600, 0.70, 263, 1.35, 1.54, 1.75, 0.53, 200, 300, 0.70, 260),

    ('CJ1', 'Tamarack'): (250.0, 51.5, 0.8025, 3.0, 0, 0.72, 2, 0.674, 41000.0, 0.21, 4.5*1.05, 0.026244,
                          0.01, 0.015, 0.011, 0.017, 0.015, 0.25, 6979.0, 8800, 10700.0, 10600.0,
                          3440.0, 100, 600, 0.70, 263, 1.4175, 1.617, 1.8375, 0.51, 180, 300, 0.70, 260),

    ('CJ1+', 'Flatwing'): (240.0, 46.5, 0.75, 0.0, 0, 0.72, 2, 0.697, 41000.0, 0.2, 4.5, 0.028,
                           0.01, 0.015, 0.011, 0.017, 0.015, 0.25, 7000.0, 8400, 10800.0, 10800.0,
                           3440.0, 100, 600, 0.70, 263, 1.35, 1.54, 1.75, 0.53, 200, 300, 0.70, 260),

    ('CJ1+', 'Tamarack'): (250.0, 51.5, 0.8025, 3.0, 0, 0.72, 2, 0.697, 41000.0, 0.21, 4.5*1.05, 0.026244,
                           0.01, 0.015, 0.011, 0.017, 0.015, 0.25, 7079.0, 8800, 10800.0, 10800.0,
                           3440.0, 100, 600, 0.70, 263, 1.4175, 1.617, 1.8375, 0.51, 180, 300, 0.70, 260),

    ('M2', 'Flatwing'): (240.0, 47.0, 0.75, 0.0, 0, 0.72, 2, 0.723, 41000.0, 0.2, 4.5, 0.028,
                         0.01, 0.015, 0.011, 0.017, 0.015, 0.25, 7000.0, 8400, 10800.0, 10700.0,
                         3440.0, 100, 600, 0.71, 263, 1.35, 1.54, 1.75, 0.53, 200, 300, 0.70, 260),

    ('M2', 'Tamarack'): (250.0, 52.0, 0.8025, 3.0, 0, 0.72, 2, 0.723, 41000.0, 0.21, 4.5*1.05, 0.026244,
                         0.01, 0.015, 0.011, 0.017, 0.015, 0.25, 7072.0, 8800, 10800.0, 10700.0,
                         3440.0, 100, 600, 0.71, 263, 1.4175, 1.617, 1.8375, 0.51, 180, 300, 0.70, 260),

    # ============================= CITATION CJ2 / CJ2+ =============================
    ('CJ2', 'Flatwing'): (264.3, 49.8, 0.75, 0.0, 0, 0.72, 2, 0.851, 45000.0, 0.2, 4.5, 0.028,
                          0.01, 0.015, 0.011, 0.017, 0.015, 0.25, 8008.0, 9800, 12500.0, 12375.0,
                          3930.0, 100, 700, 0.737, 273, 1.35, 1.54, 1.75, 0.737, 273, 800, 0.70, 260),

    ('CJ2', 'Tamarack'): (274.3, 55.3, 0.8025, 3.5, 0, 0.72, 2, 0.851, 45000.0, 0.2, 4.5, 0.026311,
                          0.01, 0.015, 0.011, 0.017, 0.015, 0.25, 8079.0, 10100, 12500.0, 12375.0,
                          3930.0, 100, 700, 0.737, 273, 1.4175, 1.617, 1.8375, 0.717, 253, 800, 0.70, 260),

    ('CJ2+', 'Flatwing'): (264.3, 49.8, 0.75, 0.0, 0, 0.72, 2, 0.883, 45000.0, 0.2, 4.5, 0.028,
                           0.01, 0.015, 0.011, 0.017, 0.015, 0.25, 8000.0, 9700, 12625.0, 12500.0,
                           3930.0, 100, 700, 0.737, 273, 1.35, 1.54, 1.75, 0.737, 273, 800, 0.70, 260),

    ('CJ2+', 'Tamarack'): (274.3, 55.3, 0.8025, 3.5, 0, 0.72, 2, 0.883, 45000.0, 0.2, 4.5, 0.026311,
                           0.01, 0.015, 0.011, 0.017, 0.015, 0.25, 8079.0, 10100, 12625.0, 12500.0,
                           3930.0, 100, 700, 0.737, 273, 1.4175, 1.617, 1.8375, 0.717, 253, 800, 0.70, 260),

    # ============================= CITATION CJ3 / CJ3+ =============================
    ('CJ3', 'Flatwing'): (294.0, 53.3, 0.75, 0.0, 0, 0.72, 2, 1.0, 45000.0, 0.2, 4.5, 0.028,
                          0.01, 0.015, 0.011, 0.017, 0.015, 0.25, 9081.0, 10510, 14070.0, 13870.0,
                          4710.0, 100, 800, 0.737, 273, 1.35, 1.54, 1.75, 0.737, 273, 800, 0.70, 260),

    ('CJ3', 'Tamarack'): (304.0, 59.3, 0.8025, 4.5, 0, 0.72, 2, 1.0, 45000.0, 0.2, 4.5, 0.026378,
                          0.01, 0.015, 0.011, 0.017, 0.015, 0.25, 9081.0, 10910, 14070.0, 13870.0,
                          4710.0, 100, 800, 0.737, 273, 1.4175, 1.617, 1.8375, 0.717, 253, 800, 0.70, 260),

    ('CJ3+', 'Flatwing'): (294.0, 53.3, 0.75, 0.0, 0, 0.72, 2, 1.0, 45000.0, 0.2, 4.5, 0.028,
                           0.01, 0.015, 0.011, 0.017, 0.015, 0.25, 9081.0, 10510, 14070.0, 13870.0,
                           4710.0, 100, 800, 0.737, 273, 1.35, 1.54, 1.75, 0.737, 273, 800, 0.70, 260),

    ('CJ3+', 'Tamarack'): (304.0, 59.3, 0.8025, 4.5, 0, 0.72, 2, 1.0, 45000.0, 0.2, 4.5, 0.026378,
                           0.01, 0.015, 0.011, 0.017, 0.015, 0.25, 9081.0, 10910, 14070.0, 13870.0,
                           4710.0, 100, 800, 0.737, 273, 1.4175, 1.617, 1.8375, 0.717, 253, 800, 0.70, 260),

    ('CJ4', 'Flatwing'): (330.0, 50.8, 0.75, 0.0, 12.5, 0.72, 2, 1.284, 45000.0, 0.2, 4.5, 0.028,
                          0.01, 0.015, 0.011, 0.017, 0.015, 0.25, 10350.0, 12500, 17230.0, 17110.0,
                          5828.0, 100, 800, 0.77, 305, 1.35, 1.54, 1.75, 0.737, 273, 800, 0.70, 260),

    ('CJ4', 'Tamarack'): (340.0, 56.3, 0.8025, 3.0, 12.5, 0.72, 2, 1.284, 45000.0, 0.2, 4.5, 0.026378,
                          0.01, 0.015, 0.011, 0.017, 0.015, 0.25, 10425.0, 12900, 17230.0, 17110.0,
                          5828.0, 100, 800, 0.77, 305, 1.4175, 1.617, 1.8375, 0.717, 253, 800, 0.70, 260),

    # ============================= CESSNA GRAND CARAVAN =============================
    ('C208B', 'Flatwing'): (279.0, 52.1, 0.80, 0.0, 0, 0.30, 1, 1.00, 25000.0, 0.2, 4.5, 0.032,
                            0.010, 0.018, 0.030, 0.020, 0.03, 0.03, 4950.0, 7000.0, 8850.0, 8750.0,
                            2200.0, 50.0, 250.0, 0.55, 175.0, 1.60, 2.20, 2.70, 0.30, 110.0, 500, 0.50, 120.0),

    ('C208B', 'Tamarack'): (299.2, 58.1, 0.84, 1.0, 0, 0.30, 1, 1.00, 25000.0, 0.2, 4.5, 0.031,
                            0.010, 0.018, 0.030, 0.020, 0.03, 0.03, 5000.0, 7050.0, 8850.0, 8750.0,
                            2200.0, 50.0, 250.0, 0.55, 175.0, 1.60, 2.20, 2.70, 0.30, 110.0, 500, 0.50, 120.0),

    ('C208EX', 'Flatwing'): (279.0, 52.1, 0.80, 0.0, 0, 0.30, 1, 1.00, 25000.0, 0.2, 4.5, 0.032,
                             0.010, 0.018, 0.030, 0.020, 0.03, 0.03, 5200.0, 7300.0, 9200.0, 9062.0,
                             2500.0, 50.0, 300.0, 0.55, 175.0, 1.60, 2.20, 2.70, 0.30, 120.0, 500, 0.50, 120.0),

    ('C208EX', 'Tamarack'): (299.2, 58.1, 0.84, 1.0, 0, 0.30, 1, 1.00, 25000.0, 0.2, 4.5, 0.031,
                             0.010, 0.018, 0.030, 0.020, 0.03, 0.03, 5250.0, 7350.0, 9200.0, 9062.0,
                             2500.0, 50.0, 300.0, 0.55, 175.0, 1.60, 2.20, 2.70, 0.30, 120.0, 500, 0.50, 120.0),

    ('C208', 'Flatwing'): (279.0, 52.1, 0.80, 0.0, 0, 0.30, 1, 1.00, 25000.0, 0.2, 4.5, 0.032,
                           0.010, 0.018, 0.030, 0.020, 0.03, 0.03, 4850.0, 7000.0, 8100.0, 8000.0,
                           2000.0, 50.0, 250.0, 0.55, 175.0, 1.60, 2.20, 2.70, 0.30, 110.0, 500, 0.50, 120.0),

    ('C208', 'Tamarack'): (299.2, 58.1, 0.84, 1.0, 0, 0.30, 1, 1.00, 25000.0, 0.2, 4.5, 0.031,
                           0.010, 0.018, 0.030, 0.020, 0.03, 0.03, 4900.0, 7050.0, 8100.0, 8000.0,
                           2000.0, 50.0, 250.0, 0.55, 175.0, 1.60, 2.20, 2.70, 0.30, 110.0, 500, 0.50, 120.0),
}

# Turboprop-specific parameters (PT6A variants in Grand Caravan family)
TURBOPROP_PARAMS = {
    'C208B': {
        'P_rated_shp': 675.0,          # PT6A-114A flat-rated to 675 shp
        'prop_diameter_ft': 8.83,      # 106 inches (common McCauley prop)
        'prop_rpm': 1900.0,
        'SSFC_lb_per_shp_hr': 0.70,     # Approximate sea-level specific fuel consumption
        'alpha_lapse': 0.60,           # Power lapse exponent with altitude
        'eta_curve_J': [0.0, 0.4, 0.8, 1.0, 1.2, 1.4],
        'eta_curve_eta': [0.00, 0.70, 0.83, 0.86, 0.82, 0.70],
        'C_T0': 0.10,
    },
    'C208EX': {
        'P_rated_shp': 867.0,          # PT6A-140, flat-rated to 867 shp
        'prop_diameter_ft': 8.83,
        'prop_rpm': 1900.0,
        'SSFC_lb_per_shp_hr': 0.68,
        'alpha_lapse': 0.60,
        'eta_curve_J': [0.0, 0.4, 0.8, 1.0, 1.2, 1.4],
        'eta_curve_eta': [0.00, 0.70, 0.84, 0.87, 0.83, 0.71],
        'C_T0': 0.10,
    },
    'C208': {
        'P_rated_shp': 675.0,
        'prop_diameter_ft': 8.83,
        'prop_rpm': 1900.0,
        'SSFC_lb_per_shp_hr': 0.70,
        'alpha_lapse': 0.60,
        'eta_curve_J': [0.0, 0.4, 0.8, 1.0, 1.2, 1.4],
        'eta_curve_eta': [0.00, 0.70, 0.83, 0.86, 0.82, 0.70],
        'C_T0': 0.10,
    }
}

# Index reference (for anyone reading the tuples)
CONFIG_INDICES = {
    'S': 0, 'b': 1, 'e': 2, 'h': 3, 'sweep_25c': 4, 'SFC': 5, 'engines': 6,
    'thrust_mult': 7, 'ceiling': 8, 'CL0': 9, 'CLA': 10, 'cdo': 11,
    'dcdo_flap1': 12, 'dcdo_flap2': 13, 'dcdo_flap3': 14, 'dcdo_gear': 15,
    'mu_to': 16, 'mu_lnd': 17, 'bow': 18, 'MZFW': 19, 'MRW': 20, 'MTOW': 21,
    'max_fuel': 22, 'taxi_fuel': 23, 'reserve_fuel': 24, 'mmo': 25, 'VMO': 26,
    'Clmax_clean': 27, 'Clmax_flaps15': 28, 'Clmax_flaps40': 29,
    'M_climb': 30, 'v_climb': 31, 'roc_min': 32, 'M_descent': 33, 'v_descent': 34
}

