import numpy as np
from scipy.interpolate import interp1d
from _src.fd_elastic_solver import solve_multi_frequency_elastic
import _src.utils as ut    

    
dir4result_save = "tmp/"
x = 60
z = 10
pml = 1
h = 0.05
ampl_sou = 1
soux = 5 + pml
souy = 0
recx = np.arange(soux+20, pml+x, 0.5)
recy = np.zeros_like(recx)

# barriers configuration
barrier_width_all = [0.2, 0.4, 0.6, 0.8, 1, 1.2]
barrier_length_all = [1.5, 3]

frequencies = np.r_[1, np.arange(5, 100, 5), np.arange(100, 260, 10)]  
nfreq = len(frequencies)
rec_indexes = np.int32(recx/h)

nx = int((x+2*pml)/h)
nz = int((z+pml)/h)
npml = int(pml/h)

# Elastic parameters for the reference model
vs_ref = 100
qp_ref = 120
puasson_ref = 0.35

# Elastic parameters for the XPS barrier 
vp_xps = 1517
vs_xps = 574
rho_xps = 41
qp_xps = interp1d([1, 50, 100, 400], [60, 40, 25, 25])(frequencies)

# Elastic parameters for the EPS barrier 
vp_eps = 392
vs_eps = 240
rho_eps = 21
qp_eps = interp1d([1, 50, 100, 400], [40, 30, 20, 20])(frequencies)

# Elastic parameters for the Rockwool barrier 
vp_rockwool = 324
vs_rockwool = 208
rho_rockwool = 150
qp_rockwool = interp1d([1, 50, 100, 400], [15, 10, 5, 5])(frequencies)

######################### Calculating displacement for a model with XPS barrier for each configuration
barrier_type = 'xps_combined'
for bar_l in barrier_length_all:
    for bar_w in barrier_width_all:
        # preparation of a reference model
        vp, vs, rho, qp, qs = ut.prepare_referent_model(nz, nx, vs_ref, puasson_ref, qp_ref, qp_ref/2, nfreq)
        # addition barrier to reference model
        vp, vs, rho, qp, qs = ut.add_barrier2reference_model(h, vp, vs, rho, qp, qs, nfreq, 
                                                             vp_xps, vs_xps, rho_xps, qp_xps, qp_xps/2, 
                                                             bar_l, bar_w, soux, bar_w/2)
        vp, vs, rho, qp, qs = ut.add_barrier2reference_model(h, vp, vs, rho, qp, qs, nfreq, 
                                                             vp_xps, vs_xps, rho_xps, qp_xps, qp_xps/2, 
                                                             bar_w, bar_l, soux+18, bar_l/2)        
        # calculation elastic equation
        data  = solve_multi_frequency_elastic(frequencies, vp, vs, qp, qs, rho, nx, nz, h, npml, souy, soux, ampl_sou)
        # saving displacement in recievers point
        ut.extract_and_save_displacement(data, nz, npml, rec_indexes, nfreq, barrier_type, bar_l, bar_w, dir4result_save)
print()

######################### Calculating displacement for a model with EPS barrier for each configuration
barrier_type = 'eps_combined'
for bar_l in barrier_length_all:
    for bar_w in barrier_width_all:
        # preparation of a reference model
        vp, vs, rho, qp, qs = ut.prepare_referent_model(nz, nx, vs_ref, puasson_ref, qp_ref, qp_ref/2, nfreq)
        # addition barrier to reference model
        vp, vs, rho, qp, qs = ut.add_barrier2reference_model(h, vp, vs, rho, qp, qs, nfreq, 
                                                             vp_eps, vs_eps, rho_eps, qp_eps, qp_eps/2, 
                                                             bar_l, bar_w, soux, bar_w/2)
        vp, vs, rho, qp, qs = ut.add_barrier2reference_model(h, vp, vs, rho, qp, qs, nfreq, 
                                                             vp_eps, vs_eps, rho_eps, qp_eps, qp_eps/2, 
                                                             bar_w, bar_l, soux+18, bar_l/2)        
        # calculation elastic equation
        data  = solve_multi_frequency_elastic(frequencies, vp, vs, qp, qs, rho, nx, nz, h, npml, souy, soux, ampl_sou)
        # saving displacement in recievers point
        ut.extract_and_save_displacement(data, nz, npml, rec_indexes, nfreq, barrier_type, bar_l, bar_w, dir4result_save)
print()

######################### Calculating displacement for a model with Rockwool barrier for each configuration
barrier_type = 'rockwool_combined'
for bar_l in barrier_length_all:
    for bar_w in barrier_width_all:
        # preparation of a reference model
        vp, vs, rho, qp, qs = ut.prepare_referent_model(nz, nx, vs_ref, puasson_ref, qp_ref, qp_ref/2, nfreq)
        # addition barrier to reference model
        vp, vs, rho, qp, qs = ut.add_barrier2reference_model(h, vp, vs, rho, qp, qs, nfreq, 
                                                             vp_rockwool, vs_rockwool, rho_rockwool, qp_rockwool, qp_rockwool/2, 
                                                             bar_l, bar_w, soux, bar_w/2)
        vp, vs, rho, qp, qs = ut.add_barrier2reference_model(h, vp, vs, rho, qp, qs, nfreq, 
                                                             vp_rockwool, vs_rockwool, rho_rockwool, qp_rockwool, qp_rockwool/2, 
                                                             bar_w, bar_l, soux+18, bar_l/2)
        # calculation elastic equation
        data  = solve_multi_frequency_elastic(frequencies, vp, vs, qp, qs, rho, nx, nz, h, npml, souy, soux, ampl_sou)
        # saving displacement in recievers point
        ut.extract_and_save_displacement(data, nz, npml, rec_indexes, nfreq, barrier_type, bar_l, bar_w, dir4result_save)
print()