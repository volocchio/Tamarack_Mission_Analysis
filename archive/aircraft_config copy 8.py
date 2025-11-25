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


AIRCRAFT_CONFIG = {
    ('CJ', 'Flatwing'): (240.0, 46.5, 0.75, 0.0, 0, 0.72, 2, 0.674, 41000.0, 0.2, 4.5, 0.027,
                         0.01, 0.015, 0.011, 0.017, 0.015, 0.25, 7200.0, 8400, 10600.0, 10500.0,
                         3440.0, 100, 600, 0.7, 263, 1.35, 1.54, 1.75, 0.53, 200, 300, 0.7, 260),
    ('CJ', 'Tamarack'): (250.0, 51.5, 0.8025, 3.0, 0, 0.72, 2, 0.674, 41000.0, 0.2, 4.5, 0.026244,
                         0.01, 0.015, 0.011, 0.017, 0.015, 0.25, 7278.65, 8800, 10600.0, 10500.0,
                         3440.0, 100, 600, 0.7, 263, 1.4175, 1.617, 1.8375, 0.51, 180, 300, 0.7, 260),
    ('CJ1', 'Flatwing'): (240.0, 46.5, 0.75, 0.0, 0, 0.72, 2, 0.674, 41000.0, 0.2, 4.5, 0.027,
                          0.01, 0.015, 0.011, 0.017, 0.015, 0.25, 7250.0, 8400, 10800.0, 10700.0,
                          3440.0, 100, 600, 0.7, 263, 1.35, 1.54, 1.75, 0.53, 200, 300, 0.7, 260),
    ('CJ1', 'Tamarack'): (250.0, 51.5, 0.8025, 3.0, 0, 0.72, 2, 0.674, 41000.0, 0.2, 4.5, 0.026244,
                          0.01, 0.015, 0.011, 0.017, 0.015, 0.25, 7328.65, 8800, 10800.0, 10700.0,
                          3440.0, 100, 600, 0.7, 263, 1.4175, 1.617, 1.8375, 0.51, 180, 300, 0.7, 260),
    ('CJ1+', 'Flatwing'): (240.0, 46.5, 0.75, 0.0, 0, 0.72, 2, 0.697, 41000.0, 0.2, 4.5, 0.027,
                           0.01, 0.015, 0.011, 0.017, 0.015, 0.25, 7250.0, 8400, 10900.0, 10800.0,
                           3440.0, 100, 600, 0.7, 263, 1.35, 1.54, 1.75, 0.53, 200, 300, 0.7, 260),
    ('CJ1+', 'Tamarack'): (250.0, 51.5, 0.8025, 3.0, 0, 0.72, 2, 0.697, 41000.0, 0.2, 4.5, 0.026244,
                           0.01, 0.015, 0.011, 0.017, 0.015, 0.25, 7328.65, 8800, 10900.0, 10800.0,
                           3440.0, 100, 600, 0.7, 263, 1.4175, 1.617, 1.8375, 0.51, 180, 300, 0.7, 260),
    ('CJ2', 'Flatwing'): (264.3, 49.8, 0.75, 0.0, 0, 0.72, 2, 0.851, 45000.0, 0.2, 4.5, 0.027,
                          0.01, 0.015, 0.011, 0.017, 0.015, 0.25, 7900.0, 9300, 12600.0, 12500.0,
                          3961.0, 100, 700, 0.737, 273, 1.35, 1.54, 1.75, 0.737, 273, 800, 0.7, 260),
    ('CJ2', 'Tamarack'): (274.3, 55.3, 0.8025, 3.5, 0, 0.72, 2, 0.851, 45000.0, 0.2, 4.5, 0.026311,
                          0.01, 0.015, 0.011, 0.017, 0.015, 0.25, 7979.26, 10100, 12600.0, 12500.0,
                          3961.0, 100, 700, 0.737, 273, 1.4175, 1.617, 1.8375, 0.717, 253, 800, 0.7, 260),
    ('CJ2+', 'Flatwing'): (264.3, 49.8, 0.75, 0.0, 0, 0.72, 2, 0.883, 45000.0, 0.2, 4.5, 0.027,
                           0.01, 0.015, 0.011, 0.017, 0.015, 0.25, 7900.0, 9700, 12725.0, 12625.0,
                           3961.0, 100, 700, 0.737, 273, 1.35, 1.54, 1.75, 0.737, 273, 800, 0.7, 260),
    ('CJ2+', 'Tamarack'): (274.3, 55.3, 0.8025, 3.5, 0, 0.72, 2, 0.883, 45000.0, 0.2, 4.5, 0.026311,
                           0.01, 0.015, 0.011, 0.017, 0.015, 0.25, 7979.26, 10100, 12725.0, 12625.0,
                           3961.0, 100, 700, 0.737, 273, 1.4175, 1.617, 1.8375, 0.717, 253, 800, 0.7, 260),
    ('CJ3', 'Flatwing'): (294.0, 53.3, 0.75, 0.0, 0, 0.72, 2, 1.0, 45000.0, 0.2, 4.5, 0.027,
                          0.01, 0.015, 0.011, 0.017, 0.015, 0.25, 8500.0, 10510, 14170.0, 14070.0,
                          4710.0, 100, 800, 0.737, 273, 1.35, 1.54, 1.75, 0.737, 273, 800, 0.7, 260),
    ('CJ3', 'Tamarack'): (304.0, 59.3, 0.8025, 4.5, 0, 0.72, 2, 1.0, 45000.0, 0.2, 4.5, 0.026378,
                          0.01, 0.015, 0.011, 0.017, 0.015, 0.25, 8580.2, 10910, 14170.0, 14070.0,
                          4710.0, 100, 800, 0.737, 273, 1.4175, 1.617, 1.8375, 0.717, 253, 800, 0.7, 260),
    ('CJ3+', 'Flatwing'): (294.0, 53.3, 0.75, 0.0, 0, 0.72, 2, 1.0, 45000.0, 0.2, 4.5, 0.027,
                           0.01, 0.015, 0.011, 0.017, 0.015, 0.25, 8500.0, 10510, 14170.0, 14070.0,
                           4710.0, 100, 800, 0.737, 273, 1.35, 1.54, 1.75, 0.737, 273, 800, 0.7, 260),
    ('CJ3+', 'Tamarack'): (304.0, 59.3, 0.8025, 4.5, 0, 0.72, 2, 1.0, 45000.0, 0.2, 4.5, 0.026378,
                           0.01, 0.015, 0.011, 0.017, 0.015, 0.25, 8580.2, 10910, 14170.0, 14070.0,
                           4710.0, 100, 800, 0.737, 273, 1.4175, 1.617, 1.8375, 0.717, 253, 800, 0.7, 260),
    ('M2', 'Flatwing'): (240.0, 47.0, 0.75, 0.0, 0, 0.72, 2, 0.723, 41000.0, 0.2, 4.5, 0.027,
                         0.01, 0.015, 0.011, 0.017, 0.015, 0.25, 8030.0, 8400, 10900.0, 10800.0,
                         3296.0, 100, 600, 0.71, 263, 1.35, 1.54, 1.75, 0.53, 200, 300, 0.7, 260),
    ('M2', 'Tamarack'): (250.0, 52.0, 0.8025, 3.0, 0, 0.72, 2, 0.723, 41000.0, 0.2, 4.5, 0.026244,
                         0.01, 0.015, 0.011, 0.017, 0.015, 0.25, 8010.0, 8800, 10900.0, 10800.0,
                         3296.0, 100, 600, 0.71, 263, 1.4175, 1.617, 1.8375, 0.51, 180, 300, 0.7, 260)
}