"""
Flight Physics Module

This module contains functions for calculating atmospheric conditions, thrust, drag,
V-speeds, and other physics-related parameters for flight simulation.
"""

import numpy as np
from math import sqrt, exp, sin


def haversine_with_bearing(lat1, lon1, lat2, lon2):
    R = 3440.065  # Nautical miles
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    distance = R * c

    y = np.sin(dlon) * np.cos(lat2_rad)
    x = np.cos(lat1_rad) * np.sin(lat2_rad) - np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(dlon)
    bearing = (np.degrees(np.arctan2(y, x)) + 360) % 360

    return distance, bearing


def atmos(alt, isa_diff):
    if alt > 36089:
        d_alt = alt + isa_diff * 96.157
    else:
        d_alt = alt + isa_diff * 118.89
    if d_alt < 36089:
        theta = 1 - (0.000006875) * d_alt
    else:
        theta = 0.7519
    if d_alt < 36089:
        sigma = theta ** 4.2621
    else:
        sigma = 0.297 * exp(-(0.00004811) * (d_alt - 36089))
    if alt < 36089:
        delta = theta ** 5.2621
    else:
        delta = 0.223 * exp(-(0.00004811) * (d_alt - 36089))
    k_temp = 288.15 * theta
    c = (1.4 * 287 * k_temp) ** 0.5 * 1.94384
    return d_alt, theta, sigma, delta, k_temp, c


def thrust_calc(d_alt, m, thrust_mult, engines, thrust_factor, segment):
    thrust_reg = (2785.75 -
                  1950.17 * m -
                  0.05261 * d_alt -
                  12.9726 * m ** 2 +
                  0.07669 * m * d_alt -
                  0.0000001806 * d_alt ** 2 +
                  1118.99 * m ** 3 -
                  0.03617 * m ** 2 * d_alt -
                  0.0000003701 * m * d_alt ** 2 +
                  0.000000000003957 * d_alt ** 3)
    thrust = thrust_reg * thrust_mult * engines * thrust_factor
    if thrust < 100:
        thrust = 100
    return thrust


def drag_calc(w, cdo, dcdo_flap1, dcdo_flap2, dcdo_flap3, dcdo_gear, m, k, cl, q, s, segment, flap):
    if m > 0.5:
        cdnp = (6.667 * m ** 4 - 15.733 * m ** 3 + 13.923 * m ** 2 - 5.464 * m + 0.8012) * (exp(6 * cl ** 2) / 4)
    else:
        cdnp = 0
    if segment in [0, 13]:
        cdi = 0
    else:
        cdi = k * cl ** 2
    if segment == 0:
        cd = cdo + cdi
    elif segment == 2:
        cd = cdo + cdi + dcdo_flap1 * flap
    elif segment == 3:
        cd = cdo + cdi
    elif segment == 11:
        cd = cdo + cdi + dcdo_flap1
    elif segment == 12:
        cd = cdo + cdi + dcdo_flap2 + dcdo_gear
    elif segment == 13:
        cd = cdo + cdi + dcdo_flap3 + dcdo_gear
    else:
        cd = cdo + cdi + cdnp
    drag = q * s * cd
    return drag, cd


def vspeeds(w, s, clmax, clmax_1, clmax_2, delta, m, flap, segment):
    try:
        # Calculate clmax_factor
        clmax_factor = 7.432 * m ** 6 - 12.59 * m ** 5 + 5.0847 * m ** 4 + 0.7356 * m ** 3 - 0.9942 * m ** 2 + 0.1147 * m + 0.9994
        
        # Calculate clmax_adjusted based on flap setting
        if flap == 0:  # Clean
            clmax_adjusted = clmax * clmax_factor
        elif flap == 1:  # Approach
            clmax_adjusted = clmax_1 * clmax_factor
        elif flap == 2:  # Landing
            clmax_adjusted = clmax_2 * clmax_factor
        else:
            clmax_adjusted = clmax * clmax_factor

        # Calculate V-speeds
        vs1g = sqrt(295 * (w / (delta * s * clmax_adjusted)))
        vs = vs1g / 1.05
        
        # Calculate VR, V1, V2, V3
        vr = 1.05 * vs
        v1 = 1.1 * vs
        v2 = 1.2 * vs
        v3 = 1.3 * vs
        
        # Calculate VAPP and VREF
        if segment in (11, 12):  # Approach and landing segments
            correction = 1 + (0.01 * (w / 1000 - 100) / 10)  # Small correction factor
            approach_denom = delta * s * clmax_1 * clmax_factor
            landing_denom = delta * s * clmax_2 * clmax_factor
            
            vapp = 20 + (1.3 * sqrt(295 * (w / approach_denom))) * correction
            vref = (1.3 * sqrt(295 * (w / landing_denom))) * correction
        else:
            vapp = None
            vref = None
            
        return vr, v1, v2, v3, vapp, vref
        
    except Exception as e:
        # Return None values to indicate calculation failure
        return None, None, None, None, None, None


def physics(t_inc, gamma, sigma, delta, w, m, c, vkias, roc_fpm, roc_goal, speed_goal, thrust_factor, engines, d_alt,
            thrust_mult, cdo, dcdo_flap1, dcdo_flap2, dcdo_flap3, dcdo_gear, k, s, segment, mu_lnd, mu_to, climb_trigger, p, mmo, v_true_fps, turboprop=None):
    drag_factor = 1
    # Turboprop thrust path: compute thrust from shaft power and prop efficiency if params provided
    if turboprop is not None:
        # Constants and parameters
        rho0 = 0.0023769  # slug/ft^3
        rho = sigma * rho0
        D = turboprop.get('prop_diameter_ft', 10.8)
        rpm = turboprop.get('prop_rpm', 1900.0)
        n = rpm / 60.0  # rev/s
        P_rated = turboprop.get('P_rated_shp', 675.0) * engines  # shp
        alpha = turboprop.get('alpha_lapse', 0.6)
        C_T0 = turboprop.get('C_T0', 0.10)
        J_curve = turboprop.get('eta_curve_J', [0.0, 0.4, 0.8, 1.0, 1.2, 1.4])
        eta_curve = turboprop.get('eta_curve_eta', [0.00, 0.70, 0.83, 0.86, 0.82, 0.70])

        # Interpolate efficiency eta(J)
        def lerp(x0, y0, x1, y1, x):
            if x1 == x0:
                return y0
            t = max(0.0, min(1.0, (x - x0) / (x1 - x0)))
            return y0 + t * (y1 - y0)

        def eta_of_J(J):
            if J <= J_curve[0]:
                return eta_curve[0]
            for i in range(1, len(J_curve)):
                if J <= J_curve[i]:
                    return lerp(J_curve[i-1], eta_curve[i-1], J_curve[i], eta_curve[i], J)
            return eta_curve[-1]

        V = max(0.0, v_true_fps)
        J = V / (n * D) if n * D > 1e-6 else 0.0
        eta = eta_of_J(J)

        # Available power with simple density lapse and throttle mapping
        P_avail_shp = P_rated * (sigma ** alpha) * max(0.0, min(1.0, thrust_factor))

        # Thrust from power, blend with static thrust at very low speeds
        T_power = (eta * P_avail_shp * 550.0) / max(V, 1e-3)  # 1 shp = 550 ft*lbf/s
        T_static = C_T0 * rho * (n ** 2) * (D ** 4)
        # Blend region between 10 and 80 ft/s
        V_lo, V_hi = 10.0, 80.0
        if V <= V_lo:
            thrust = T_static
        elif V >= V_hi:
            thrust = T_power
        else:
            w_blend = (V - V_lo) / (V_hi - V_lo)
            thrust = (1 - w_blend) * T_static + w_blend * T_power
        if thrust < 100:
            thrust = 100
    else:
        thrust = thrust_calc(d_alt, m, thrust_mult, engines, thrust_factor, segment)

    if p == 0:
        m = 0

    vkeas = vkias / (1 + 1/8 * (1 - delta) * m ** 2 + 3/640 * (1 - 10 * delta + 9 * delta ** 2) * m ** 4)
    q = vkeas ** 2 / 295
    if segment == 0:
        cl = 0
    else:
        cl = 0 if q == 0 else w / (q * s)
        if cl > 2.0:
            cl = 2.0
    drag, cd = drag_calc(w, cdo, dcdo_flap1, dcdo_flap2, dcdo_flap3, dcdo_gear, m, k, cl, q, s, segment, 1)
    drag *= drag_factor
    if segment == 0:
        drag_gnd = mu_to * w
    elif segment == 13:
        drag_gnd = mu_lnd * w
    else:
        drag_gnd = 0
    acc_x = 32.2 * (thrust - (drag + drag_gnd)) / w

    if speed_goal < 1 and segment != 13:
        target_v_fps = speed_goal * c * (6076.12 / 3600)
    else:
        correction = 1 + 1/8 * (1 - delta) * m ** 2 + 3/640 * (1 - 10 * delta + 9 * delta ** 2) * m ** 4
        vkeas = speed_goal / correction
        vktas = vkeas / sqrt(sigma)
        target_v_fps = vktas * (6076.12 / 3600)

    if v_true_fps < target_v_fps:
        if v_true_fps + acc_x * t_inc > target_v_fps:
            acc_x = (target_v_fps - v_true_fps) / t_inc
    elif segment == 5 or (v_true_fps > target_v_fps and segment != 13):
        if v_true_fps + acc_x * t_inc > target_v_fps:
            acc_x = (target_v_fps - v_true_fps) / t_inc
            if acc_x < -3 and segment != 13:
                acc_x = -3
                climb_trigger = 1
            else:
                climb_trigger = 0
    elif segment in [8, 9, 10, 11, 12, 13]:
        if v_true_fps + acc_x * t_inc > target_v_fps:
            acc_x = (target_v_fps - v_true_fps) / t_inc
            if acc_x < -3 and segment != 13:
                acc_x = -3
            if segment == 13:
                acc_x = 32.2 * (thrust - (drag + drag_gnd)) / w
                v_true_fps = v_true_fps + acc_x * t_inc
    else:
        acc_x = 0
        thrust = drag

    v_true_fps += acc_x * t_inc
    if v_true_fps < 0:
        v_true_fps = 0
    vktas = v_true_fps / (6076.12 / 3600)
    m = vktas / c
    if m > mmo:
        m = mmo
    vkeas = vktas * sqrt(sigma)
    correction = 1 + 1/8 * (1 - delta) * m ** 2 + 3/640 * (1 - 10 * delta + 9 * delta ** 2) * m ** 4
    if correction > 1000:
        correction = 1000
    vkias = vkeas * correction
    return cl, q, drag, cd, vkeas, vktas, v_true_fps, thrust, drag_gnd, vkias, m


def predict_roc(next_step_alt, alt, w, m, thrust, drag, vktas, thrust_mult, engines, thrust_factor, cdo, dcdo_flap1, dcdo_flap2, dcdo_flap3, dcdo_gear, k, s, isa_diff, speed_goal, segment, alt_goal):
    new_d_alt, new_theta, new_sigma, new_delta, new_k_temp, new_c = atmos(next_step_alt, isa_diff)
    if speed_goal < 1:
        new_m = speed_goal
        new_vktas = new_m * new_c
        new_vkeas = new_vktas * sqrt(new_sigma)
    else:
        correction = 1 + 1/8 * (1 - new_delta) * m ** 2 + 3/640 * (1 - 10 * new_delta + 9 * new_delta ** 2) * m ** 4
        new_vkeas = speed_goal / correction
        new_vktas = new_vkeas / sqrt(new_sigma)
        new_m = new_vktas / new_c
    v_true_fps_new = new_vktas * 6076.12 / 3600
    new_thrust = thrust_calc(new_d_alt, new_m, thrust_mult, engines, thrust_factor, segment)
    new_q = new_vkeas ** 2 / 295
    if new_q <= 0:
        new_cl = 0
        new_drag = 0
    else:
        new_cl = w / (new_q * s)
        new_drag, new_cd = drag_calc(w, cdo, dcdo_flap1, dcdo_flap2, dcdo_flap3, dcdo_gear, new_m, k, new_cl, new_q, s, segment, 1)
    new_tx = (new_thrust - new_drag) / w
    new_gamma = np.arcsin(new_tx) if -1 <= new_tx <= 1 else (np.pi/2 if new_tx > 1 else -np.pi/2)
    new_roc_fps = new_gamma * v_true_fps_new / (6076.12 / 3600)
    new_roc_fpm = new_roc_fps * 60
    if new_roc_fpm < 0:
        new_roc_fpm = 0
    return new_roc_fpm


def next_step_altitude(current_alt, final_cruise_alt, previous_step_alt):
    rounded_current_alt = round(current_alt / 1000) * 1000
    rounded_final_cruise_alt = round(final_cruise_alt / 1000) * 1000
    if current_alt >= rounded_final_cruise_alt:
        return rounded_final_cruise_alt
    if previous_step_alt > 0 and current_alt < previous_step_alt:
        return previous_step_alt
    final_is_even = (rounded_final_cruise_alt // 1000) % 2 == 0
    if final_is_even:
        if (rounded_current_alt // 1000) % 2 != 0:
            next_alt = rounded_current_alt + 1000
        else:
            next_alt = rounded_current_alt + 2000
    else:
        if (rounded_current_alt // 1000) % 2 == 0:
            next_alt = rounded_current_alt + 1000
        else:
            next_alt = rounded_current_alt + 2000
    if next_alt > rounded_final_cruise_alt:
        next_alt = rounded_final_cruise_alt
    return round(next_alt / 1000) * 1000