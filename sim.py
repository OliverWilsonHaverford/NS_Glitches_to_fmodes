# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 11:26:58 2025

@author: olive
"""

# Stabilization and dynamics of superfluid vortices in a neutron star
# Ported from C++ by Oliver Wilson with the help of ChatGPT

import os
import math
import random
import time
from pathlib import Path
import numpy as np

# ----- Shorthands are removed in Python (use functions instead) -----

# ----- Directory for outputs -----
path = "./InOut/"  # Make sure this directory exists
pathplace = ""
numb_runs = 5
run_id = 1

# ----- Constants and parameters -----

pi = math.pi

# Basic parameters
nv = 2000
npin_desired = 20000
numb_trig = 0
R = 10
k = 1.0
phi = 0.1
V_1 = 2000
V_2 = 2000
trig_duration = 0
decel = -0.25e-3
stab_time = 100
sim_time = 2000
trig_type = "stresswave"

# Advanced parameters
pin_config = "annular"
trig_region = "fullsector"
write_fullpos = False
k_img = k
omega_0 = (nv * k) / (R * R)
T_0 = (2 * pi) / omega_0
a = R * math.sqrt(pi / npin_desired)
Xi = 0.1 * a
N_ext_0 = decel * (omega_0 / T_0)
n_ratio = 1
r_annulus = R / math.sqrt(2)
duration_trig = trig_duration * T_0
runtime_stabilization = stab_time * T_0
runtime_simulation = sim_time * T_0
del_t = 5e-3
N_ext = N_ext_0
omega_c = omega_0
omega_s = omega_0
sector_count = 8
stresswave_threshold = 0.7
epsilon_glitch = 1e-12

# ----- Initial variables -----

npin = 0
vortex_out_count = 0
nmax = int(math.floor(2 * R / a) + 1)
type_counter = 0
trial_count = 0
count_trig = 0
state_trig = 0
unpinned_count = 0
vortex_in_count = 0
H = 0
kiss = 0
prog = 0
time_start_trig = 0
t = 0

# ----- Initial vectors -----

f = []  # state vector

vortex_vectx = [0.0] * nv
vortex_vecty = [0.0] * nv
imgvortex_vectx = [0.0] * nv
imgvortex_vecty = [0.0] * nv
K = [k] * nv
K_img = [k_img] * nv
omega_s_vector = [omega_0]
omega_c_vector = [omega_0]
V_0 = [[0.0 for _ in range(nmax)] for _ in range(nmax)]
V_0_original = [[0.0 for _ in range(nmax)] for _ in range(nmax)]
times_trig = [0.0] * numb_trig
omega_s_forsum = [0.0] * nv

# Backups for reinitialization
i_vortex_vectx = list(vortex_vectx)
i_vortex_vecty = list(vortex_vecty)
i_imgvortex_vectx = list(imgvortex_vectx)
i_imgvortex_vecty = list(imgvortex_vecty)
i_K = list(K)
i_K_img = list(K_img)
i_omega_s_vector = list(omega_s_vector)
i_omega_c_vector = list(omega_c_vector)
i_V_0 = [row[:] for row in V_0]
i_V_0_original = [row[:] for row in V_0_original]
i_times_trig = list(times_trig)
i_omega_s_forsum = list(omega_s_forsum)

# Output tracking
Hvalues = []
tvalues = []
vortex_states = []
omega_c_vec = []
omega_s_vec = []
numboff_vec = []
sector_vec = []
unpinned_vec = []
vortex_in_vec = []
#Simulation loop
def run_simulation():
    global run_id

    for run_id in range(1, numb_runs + 1):
        print(f"Run {run_id} of {numb_runs}")

        # Initial resets
        reinitialize()

        # Create output directory and initialize files
        create_output_files(run_id)
        vortexinit1()
        initial_state()
        circulation_init()
        vortex_counter()
        pin_config_set()
        pins_count()
        write_info_start()
        stabilization()
        write_data_post_stabilization()
        t = 0
        tvalues.clear()
        prog = 0
        trig_init()
        vortex_counter()
        kiss_fix()
        write_info_mid()
        simulation()
        write_info_end()
        write_data_post_simulation()
        find_glitch(omega_c_vec, tvalues, epsilon_glitch)

        print("Done with run.\n")
#Reinitializing variables function
def reinitialize():
    global N_ext, omega_c, omega_s
    global npin, vortex_out_count, nmax, type_counter, trial_count
    global count_trig, state_trig, unpinned_count, vortex_in_count
    global H, kiss, prog, time_start_trig, t
    global f, vortex_vectx, vortex_vecty, imgvortex_vectx, imgvortex_vecty
    global K, K_img, omega_s_vector, omega_c_vector
    global V_0, V_0_original, times_trig, omega_s_forsum
    global Hvalues, tvalues, vortex_states
    global omega_c_vec, omega_s_vec, numboff_vec
    global sector_vec, unpinned_vec, vortex_in_vec

    # Scalar resets
    N_ext = N_ext_0
    omega_c = omega_0
    omega_s = omega_0

    npin = 0
    vortex_out_count = 0
    nmax = int(math.floor(2 * R / a) + 1)
    type_counter = 0
    trial_count = 0
    count_trig = 0
    state_trig = 0
    unpinned_count = 0
    vortex_in_count = 0
    H = 0
    kiss = 0
    prog = 0
    time_start_trig = 0
    t = 0

    # Vector resets
    vortex_vectx = list(i_vortex_vectx)
    vortex_vecty = list(i_vortex_vecty)
    imgvortex_vectx = list(i_imgvortex_vectx)
    imgvortex_vecty = list(i_imgvortex_vecty)
    K = list(i_K)
    K_img = list(i_K_img)
    omega_s_vector = list(i_omega_s_vector)
    omega_c_vector = list(i_omega_c_vector)
    V_0 = [row[:] for row in i_V_0]
    V_0_original = [row[:] for row in i_V_0_original]
    times_trig = list(i_times_trig)
    omega_s_forsum = list(i_omega_s_forsum)

    # Output vectors
    f.clear()
    Hvalues.clear()
    tvalues.clear()
    vortex_states.clear()
    omega_c_vec.clear()
    omega_s_vec.clear()
    numboff_vec.clear()
    sector_vec.clear()
    unpinned_vec.clear()
    vortex_in_vec.clear()
#output files
def create_output_files(run_id):
    global pathplace
    var = str(run_id)
    pathplace = os.path.join(path, f"run{var}/")
    Path(pathplace).mkdir(parents=True, exist_ok=True)

    def init_file(filename, header=None):
        fullpath = os.path.join(pathplace, filename)
        with open(fullpath, 'w') as f:
            if header:
                f.write(header + '\n')

    init_file("sim_vortex_pos.dat", "t/T_0\t" + "\t".join([f"x{i+1}" for i in range(nv)] + [f"y{i+1}" for i in range(nv)]))
    init_file("sim_vortex_pos_minimal.dat", "t/T_0\t" + "\t".join([f"x{i+1}" for i in range(nv)] + [f"y{i+1}" for i in range(nv)]))
    init_file("init_vortex_pos.dat", "x\ty")
    init_file("init_img_pos.dat", "x\ty")
    init_file("stabilized_Hvalues.dat", "t/T_0\tH")
    init_file("stabilized_vortex_pos.dat")
    init_file("sim_omega_c.dat", "t/T_0\tomega_c/omega_0")
    init_file("sim_omega_s.dat", "t/T_0\tomega_s/omega_0")
    init_file("info_simulation.dat")
    init_file("info_progress.dat")
    init_file("sim_triggers.dat", "t/T_0\tsector_id\tnumboff" if trig_type == "sectorial" else "t/T_0")
    init_file("sim_unpinned.dat", "t/T_0\tunpinned_count\ttotal_count\tunpinned_frac")
    init_file("info_glitch.dat")
    init_file("data_glitch.dat")
def write_progress(runtime, phase_label="Phase", step_fraction=0.01):
    global prog
    percent_done = ((t / runtime) * 100)

    if percent_done >= prog + 0.01:
        prog = percent_done
        msg = f"{phase_label} progress: {prog}% complete"
        print(msg)  # Console output
        with open(os.path.join(pathplace, "info_progress.dat"), "a") as f:
            f.write(msg + "\n")
def vortexinit1():
    global vortex_vectx, vortex_vecty, imgvortex_vectx, imgvortex_vecty

    for i in range(nv):
        r = R * math.sqrt(random.uniform(0, 1))
        theta = 2 * pi * random.uniform(0, 1)

        x = r * math.cos(theta)
        y = r * math.sin(theta)

        vortex_vectx[i] = x
        vortex_vecty[i] = y

        r_img = R**2 / r
        x_img = r_img * math.cos(theta)
        y_img = r_img * math.sin(theta)

        imgvortex_vectx[i] = x_img
        imgvortex_vecty[i] = y_img

    # Write to file
    with open(os.path.join(pathplace, "init_vortex_pos.dat"), "a") as fout_v, \
         open(os.path.join(pathplace, "init_img_pos.dat"), "a") as fout_i:
        for i in range(nv):
            fout_v.write(f"{vortex_vectx[i]:.15f}\t{vortex_vecty[i]:.15f}\n")
            fout_i.write(f"{imgvortex_vectx[i]:.15f}\t{imgvortex_vecty[i]:.15f}\n")
def initial_state():
    global f
    f = vortex_vectx + vortex_vecty  # concatenate x and y positions
def circulation_init():
    global K, K_img

    for i in range(nv):
        r = math.hypot(f[i], f[i + nv])
        if r > R:
            K[i] = 0
            K_img[i] = 0
        else:
            K[i] = k
            K_img[i] = k
def vortex_counter():
    global vortex_out_count, vortex_in_count, unpinned_count

    vortex_out_count = 0
    vortex_in_count = 0
    unpinned_count = 0

    for i in range(nv):
        xi = f[i]
        yi = f[i + nv]
        r = math.hypot(xi, yi)

        if r >= R:
            vortex_out_count += 1
        else:
            vortex_in_count += 1

            x_pin_nearest = -R + round((xi + R) / a) * a
            y_pin_nearest = -R + round((yi + R) / a) * a
            r_pin_nearest = math.hypot(x_pin_nearest - xi, y_pin_nearest - yi)

            if r_pin_nearest > Xi:
                unpinned_count += 1
def pin_strength_initialize_annular():
    global V_0
    for x in np.arange(-R, R + a, a):
        for y in np.arange(-R, R + a, a):
            dist_pin = math.hypot(x, y)
            if dist_pin < R:
                nx = round((R + x) / a)
                ny = round((R + y) / a)
                if dist_pin <= r_annulus:
                    V_0[nx][ny] = V_1
                else:
                    V_0[nx][ny] = V_2
def pin_strength_initialize_half():
    global V_0
    for x in np.arange(-R, R + a, a):
        for y in np.arange(-R, R + a, a):
            dist_pin = math.hypot(x, y)
            if dist_pin < R:
                nx = round((R + x) / a)
                ny = round((R + y) / a)
                V_0[nx][ny] = V_1 if x < 0 else V_2
def pin_strength_initialize_alternate():
    global V_0, type_counter
    type_counter = 0
    for x in np.arange(-R, R + a, a):
        for y in np.arange(-R, R + a, a):
            dist_pin = math.hypot(x, y)
            if dist_pin < R:
                nx = round((R + x) / a)
                ny = round((R + y) / a)
                if type_counter != n_ratio:
                    V_0[nx][ny] = V_1
                    type_counter += 1
                else:
                    V_0[nx][ny] = V_2
                    type_counter = 0
def pin_config_set():
    global V_0_original

    if pin_config == "half":
        pin_strength_initialize_half()
    elif pin_config == "alternate":
        pin_strength_initialize_alternate()
    elif pin_config == "annular":
        pin_strength_initialize_annular()
    
    # Save original for reset after trigger
    V_0_original = [row[:] for row in V_0]
def pins_count():
    global npin
    npin = 0
    for x in np.arange(-R, R + a, a):
        for y in np.arange(-R, R + a, a):
            if math.hypot(x, y) < R:
                npin += 1
def write_info_start():
    with open(os.path.join(pathplace, "info_simulation.dat"), "a") as f:
        f.write(f"nv = {nv}\n")
        f.write(f"npin = {npin}\n")
        f.write(f"pin_config = {pin_config}\n")
        f.write(f"r_annulus = {r_annulus / R:.4f} R\n")
        f.write(f"R = {R}\n")
        f.write(f"omega_0 = {omega_0}\n")
        f.write(f"T_0 = {T_0}\n")
        f.write(f"phi = {phi}\n")
        f.write(f"V_1 = {V_1}\n")
        f.write(f"V_2 = {V_2}\n")
        f.write(f"n_ratio = {n_ratio}\n")
        f.write(f"a = {a / R:.4f} R\n")
        f.write(f"Xi = {Xi / R:.4f} R\n")
        f.write(f"N_ext = {N_ext * (T_0 / omega_0):.6f} omega_0/T_0\n")
        f.write(f"initial del_t = {del_t / T_0:.6f} T_0\n\n")
        f.write(f"trig_region = {trig_region}\n")
        f.write(f"numb_trig = {numb_trig}\n")
        f.write(f"duration_trig = {duration_trig / T_0:.6f} T_0\n")
        f.write(f"sector_count = {sector_count}\n")
        f.write(f"trig_type = {trig_type}\n")
        f.write(f"Stabilization runtime = {runtime_stabilization / T_0:.2f} T_0\n\n")
        f.write(f"Simulation runtime = {runtime_simulation / T_0:.2f} T_0\n\n")
        f.write(f"Start time = {time.ctime()}\n")
        f.write(f"nv_out = {vortex_out_count}\n\n")
def stabilization():
    global t
    while t <= runtime_stabilization:
        write_progress(runtime_stabilization, "Stabilization")
        tvalues.append(t / T_0)
        H_calculate_save()

        integ_adaptive()
        t += del_t

        updates(omega_s_condition=0)  # update image vortices + circulation (no omega_s in stab)
def H_calculate_save():
    global H, Hvalues
    H_forsum = []

    for i in range(nv):
        h_i = 0
        xi, yi = f[i], f[i + nv]
        for j in range(nv):
            if j != i:
                xj, yj = f[j], f[j + nv]
                rij = math.hypot(xi - xj, yi - yj)
                h_i += math.log(rij)
        H_forsum.append(h_i)

    H = sum(H_forsum)
    Hvalues.append(H)
from scipy.integrate import solve_ivp

def integ_adaptive():
    global f
    t_span = (t, t + del_t)
    f0 = np.array(f)

    sol = solve_ivp(fun=eom_wrapper, t_span=t_span, y0=f0, method="RK45", rtol=1e-5, atol=1e-8)

    if sol.success:
        f[:] = sol.y[:, -1].tolist()
    else:
        print("Integration failed.")
def eom_wrapper(t, f_local):
    dfdt = [0.0] * (2 * nv)

    for i in range(nv):
        xi = f_local[i]
        yi = f_local[i + nv]

        if K[i] == 0:
            continue

        dxdt = omega_c * yi
        dydt = -omega_c * xi
        Ki = K[i]

        for j in range(nv):
            if i == j or K[j] == 0:
                continue
            xj, yj = f_local[j], f_local[j + nv]
            dx = xi - xj
            dy = yi - yj
            r2 = dx**2 + dy**2
            dxdt += -K[j] * Ki * dy / r2
            dydt += K[j] * Ki * dx / r2

            dx_img = xi - imgvortex_vectx[j]
            dy_img = yi - imgvortex_vecty[j]
            r2_img = dx_img**2 + dy_img**2
            dxdt += K_img[j] * Ki * dy_img / r2_img
            dydt += -K_img[j] * Ki * dx_img / r2_img

        nx = int(round((xi + R) / a))
        ny = int(round((yi + R) / a))
        if nx <= 0 or ny <= 0 or nx >= nmax or ny >= nmax:
            nx, ny = 0, 0

        x_pin = -R + round((xi + R) / a) * a
        y_pin = -R + round((yi + R) / a) * a
        r2_pin = (x_pin - xi)**2 + (y_pin - yi)**2
        dxdt += V_0[nx][ny] * math.exp(-r2_pin / (2 * Xi**2)) * (yi - y_pin)
        dydt += -V_0[nx][ny] * math.exp(-r2_pin / (2 * Xi**2)) * (xi - x_pin)

        # Apply dissipation
        vel_x = dxdt * K[i]
        vel_y = dydt * K[i]
        dfdt[i] = math.cos(phi) * vel_x + math.sin(phi) * vel_y
        dfdt[i + nv] = -math.sin(phi) * vel_x + math.cos(phi) * vel_y

    return dfdt
def updates(omega_s_condition):
    for i in range(nv):
        if K[i] == 0:
            continue

        xi = f[i]
        yi = f[i + nv]
        r = math.hypot(xi, yi)
        theta = math.atan2(yi, xi)

        r_img = R**2 / r
        imgvortex_vectx[i] = r_img * math.cos(theta)
        imgvortex_vecty[i] = r_img * math.sin(theta)

        if r > R:
            K[i] = 0
            K_img[i] = 0
        else:
            K[i] = k
            K_img[i] = k

        if omega_s_condition == 1:
            omega_s_forsum[i] = K[i] * kiss * (R**2 - r**2)

    if omega_s_condition == 1:
        global omega_s
        omega_s = sum(omega_s_forsum)
        omega_s_vector.append(omega_s)
        vortex_counter()
def write_data_post_stabilization():
    # Write final vortex positions
    with open(os.path.join(pathplace, "stabilized_vortex_pos.dat"), "a") as f_pos:
        f_pos.write("x\ty\n")
        for i in range(nv):
            f_pos.write(f"{f[i]:.15f}\t{f[i + nv]:.15f}\n")
        f_pos.write("\n")

    # Write H values vs time
    with open(os.path.join(pathplace, "stabilized_Hvalues.dat"), "a") as f_H:
        for t_val, H_val in zip(tvalues, Hvalues):
            f_H.write(f"{t_val:.15f}\t{H_val:.15f}\n")
def trig_init():
    global times_trig
    times_trig = sorted([random.uniform(0, runtime_simulation) for _ in range(numb_trig)])
    times_trig.append(runtime_simulation + 5)  # ensures no trigger happens past sim time
def kiss_fix():
    global kiss
    total = 0
    for i in range(nv):
        xi = f[i]
        yi = f[i + nv]
        r = math.hypot(xi, yi)
        if r < R:
            total += R**2 - r**2
    kiss = omega_0 / total if total != 0 else 0
def write_info_mid():
    with open(os.path.join(pathplace, "info_simulation.dat"), "a") as f:
        f.write(f"Stabilization end time = {time.ctime()}\n\n")
        f.write("Simulation begins.\n")
        f.write(f"nv_out = {vortex_out_count}\n")
        f.write(f"unpinned_count = {unpinned_count}\n")
        f.write(f"omega_c = {omega_c / omega_0:.6f} omega_0\n\n")
        f.write(f"k/I_s = {kiss:.6f}\n\n\n")
def simulation():
    global t, omega_c

    while t <= runtime_simulation:
        write_progress(runtime_simulation, "Simulation")

        tvalues.append(t / T_0)
        vortex_states.append(list(f))  # copy current state
        omega_c_vec.append(omega_c / omega_0)
        omega_s_vec.append(omega_s / omega_0)
        unpinned_vec.append(unpinned_count)
        vortex_in_vec.append(vortex_in_count)

        # Trigger handling
        if trig_duration != 0:
            trigger_check(duration_trig, t)

        # Crust spindown
        omega_c += N_ext * del_t

        # Integrate system
        integ_adaptive()
        t += del_t

        # Update circulations, image vortices, etc.
        updates(omega_s_condition=1)

        # Feedback from superfluid to crust
        if len(omega_s_vector) >= 2:
            delta_omega_s = omega_s_vector[-1] - omega_s_vector[-2]
            omega_c -= delta_omega_s
def write_info_end():
    with open(os.path.join(pathplace, "info_simulation.dat"), "a") as f:
        f.write(f"End time = {time.ctime()}\n")
        f.write(f"nv_out = {vortex_out_count}\n")
        f.write(f"omega_c = {omega_c / omega_0:.6f} omega_0\n\n\n")
def write_vortex_positions():
    # Full position data (optional based on flag)
    if write_fullpos:
        with open(os.path.join(pathplace, "sim_vortex_pos.dat"), "a") as f:
            for i in range(len(tvalues)):
                pos = "\t".join(f"{x:.15f}" for x in vortex_states[i])
                f.write(f"{tvalues[i]:.15f}\t{pos}\n")

    # Minimal (every 10th timestep)
    with open(os.path.join(pathplace, "sim_vortex_pos_minimal.dat"), "a") as f:
        for i in range(0, len(tvalues), 10):
            pos = "\t".join(f"{x:.15f}" for x in vortex_states[i])
            f.write(f"{tvalues[i]:.15f}\t{pos}\n")
def write_omega_series():
    with open(os.path.join(pathplace, "sim_omega_c.dat"), "a") as f_c, \
         open(os.path.join(pathplace, "sim_omega_s.dat"), "a") as f_s:
        for t_val, oc, os in zip(tvalues, omega_c_vec, omega_s_vec):
            f_c.write(f"{t_val:.15f}\t{oc:.15f}\n")
            f_s.write(f"{t_val:.15f}\t{os:.15f}\n")
def write_triggers():
    filepath = os.path.join(pathplace, "sim_triggers.dat")
    with open(filepath, "a") as f:
        if trig_type == "sectorial":
            for t_val, sid, off in zip(times_trig[:-1], sector_vec, numboff_vec):
                f.write(f"{t_val / T_0:.15f}\t{sid}\t{off}\n")
        elif trig_type == "stresswave":
            for t_val in times_trig[:-1]:
                f.write(f"{t_val / T_0:.15f}\n")
def write_unpinned():
    with open(os.path.join(pathplace, "sim_unpinned.dat"), "a") as f:
        for t_val, unpinned, inside in zip(tvalues, unpinned_vec, vortex_in_vec):
            frac = unpinned / inside if inside != 0 else 0
            f.write(f"{t_val:.15f}\t{unpinned}\t{inside}\t{frac:.6f}\n")
def write_data_post_simulation():
    write_vortex_positions()
    write_omega_series()
    write_triggers()
    write_unpinned()
def trigger_check(duration_trig, t_curr):
    global time_start_trig, state_trig, count_trig, V_0

    if trig_type == "stresswave":
        if t_curr > times_trig[count_trig]:
            time_start_trig = t_curr
            state_trig = 1
            count_trig += 1
            pin_reduce()

        if state_trig == 1 and t_curr > time_start_trig + duration_trig:
            state_trig = 0
            V_0 = [row[:] for row in V_0_original]

    elif trig_type == "sectorial":
        if t_curr > times_trig[count_trig]:
            time_start_trig = t_curr
            state_trig = 1
            count_trig += 1
            pin_off()

        if state_trig == 1 and t_curr > time_start_trig + duration_trig:
            state_trig = 0
            V_0 = [row[:] for row in V_0_original]
def pin_off():
    numboff = 0
    sector_id = int(random.uniform(0, sector_count))
    sector_vec.append(sector_id)

    for x in np.arange(-R, R + a, a):
        for y in np.arange(-R, R + a, a):
            dist_pin = math.hypot(x, y)
            if dist_pin >= R:
                continue

            nx = round((R + x) / a)
            ny = round((R + y) / a)

            if trig_region == "outer" and dist_pin <= r_annulus:
                continue
            if trig_region == "inner" and dist_pin > r_annulus:
                continue

            if sector_check(x, y, sector_id) and random.random() <= 0.5:
                V_0[nx][ny] = 0
                numboff += 1

    numboff_vec.append(numboff)
def pin_reduce():
    for x in np.arange(-R, R + a, a):
        for y in np.arange(-R, R + a, a):
            if math.hypot(x, y) >= R:
                continue
            nx = round((R + x) / a)
            ny = round((R + y) / a)
            V_0[nx][ny] = stresswave_threshold * V_1
def sector_check(x, y, sector_id):
    angle = math.degrees(math.atan2(y if y != 0 else 1e-6, x if x != 0 else 1e-6))
    if angle < 0:
        angle += 360

    angle_start = sector_id * (360 / sector_count)
    angle_end = angle_start + (360 / sector_count)

    return angle_start < angle < angle_end
def find_glitch(omega_vec, t_vec, epsilon):
    glitch_state = False
    glitch_init = False

    t_i = omega_i = t_f = omega_f = 0
    t_glitch = []
    t_rise = []
    del_omega = []

    for i in range(1, len(t_vec)):
        omegadot = (omega_vec[i] - omega_vec[i - 1]) / (t_vec[i] - t_vec[i - 1])
        if not glitch_state and omegadot > 0:
            glitch_state = True
            glitch_init = True
        elif glitch_state and omegadot < 0:
            glitch_state = False
            t_f = t_vec[i - 1]
            omega_f = omega_vec[i - 1]
            delta = omega_f - omega_i
            if delta > epsilon:
                t_glitch.append(t_i)
                t_rise.append(t_f - t_i)
                del_omega.append(delta)

        if glitch_init:
            t_i = t_vec[i - 1]
            omega_i = omega_vec[i - 1]
            glitch_init = False

    # Waiting times
    t_wait = [t_glitch[i] - t_glitch[i - 1] for i in range(1, len(t_glitch))]

    # Remove first glitch for uniformity
    if len(t_glitch) > 1:
        t_glitch = t_glitch[1:]
        t_rise = t_rise[1:]
        del_omega = del_omega[1:]

    # Stats
    if del_omega:
        mean = sum(del_omega) / len(del_omega)
        median = np.median(del_omega)
        max_val = max(del_omega)
        min_val = min(del_omega)
    else:
        mean = median = max_val = min_val = 0

    # Write info
    with open(os.path.join(pathplace, "info_glitch.dat"), "a") as f:
        f.write(f"Smallest allowed glitch size = {epsilon} omega_0\n")
        f.write(f"{len(t_glitch)} glitches found\n")
        f.write(f"Biggest glitch size = {max_val:.5e} omega_0\n")
        f.write(f"Smallest glitch size = {min_val:.5e} omega_0\n")
        f.write(f"Mean glitch size = {mean:.5e} omega_0\n")
        f.write(f"Median glitch size = {median:.5e} omega_0\n")

    with open(os.path.join(pathplace, "data_glitch.dat"), "a") as f:
        f.write("t_glitch\tt_rise\tglitch_size\tt_wait\n")
        for i in range(len(t_glitch)):
            wait = t_wait[i] if i < len(t_wait) else 0
            f.write(f"{t_glitch[i]:.15f}\t{t_rise[i]:.15f}\t{del_omega[i]:.15f}\t{wait:.15f}\n")
os.makedirs(path, exist_ok=True)
run_simulation()