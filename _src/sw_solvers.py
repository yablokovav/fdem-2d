import numpy as np
from disba import DispersionError, GroupDispersion, PhaseDispersion, EigenFunction
from joblib import Parallel, delayed
np.complex_ = np.complex128
DC_VALUE = 5e-05 # Phase velocity increment for root finding.

def eigenfunctions_solver(
    num_mode: int,
    f: np.ndarray,
    vs: np.ndarray,
    vp: np.ndarray,
    rho: np.ndarray,
    thk: np.ndarray,
):
 
    # Construct the velocity model array expected by disba
    velocity_model: np.ndarray = (
        np.vstack(
            (
                np.hstack((np.squeeze(thk), 0))[None, :],  # Layer thicknesses (with half-space) in km
                np.atleast_2d(vp),  # Compressional wave velocities in km/s
                np.atleast_2d(vs),  # Shear wave velocities in km/s
                np.atleast_2d(rho),  # Densities in g/cm^3
            )
        )
        / 1000  # Convert units to km/s and g/cm^3
    )

    t: np.ndarray = np.atleast_1d(1 / f)  # Calculate periods (s) from frequencies (Hz)
    ur: np.ndarray = np.zeros((t.size, vs.size))
    uz: np.ndarray = np.zeros((t.size, vs.size))
    # uu: np.ndarray = np.zeros((t.size, vs.size))
    
    solver_class = EigenFunction(*velocity_model)
    for ifreq in range(t.size):
        # try:
        #     uu[ifreq, :] = solver_class(t[ifreq], mode = num_mode, wave = "love").uu
        # except Exception:
        #     uu[ifreq, :] = np.nan
        try:
            eig = solver_class(t[ifreq], mode = num_mode, wave = "rayleigh")
            ur[ifreq, :] = eig.ur
            uz[ifreq, :] = eig.uz
        except Exception:
            ur[ifreq, :] = np.nan
            uz[ifreq, :] = np.nan
    return uz, ur

def dc_solver(
    num_mode: int,
    f: np.ndarray,
    vs: np.ndarray,
    vp: np.ndarray,
    rho: np.ndarray,
    thk: np.ndarray,
    wave: "WaveType" = "rayleigh",
    veltype: "VelocityType" = "phase",
):
 
    # Construct the velocity model array expected by disba
    velocity_model: np.ndarray = (
        np.vstack(
            (
                np.hstack((np.squeeze(thk), 0))[None, :],  # Layer thicknesses (with half-space) in km
                np.atleast_2d(vp),  # Compressional wave velocities in km/s
                np.atleast_2d(vs),  # Shear wave velocities in km/s
                np.atleast_2d(rho),  # Densities in g/cm^3
            )
        )
        / 1000  # Convert units to km/s and g/cm^3
    )

    t: np.ndarray = 1 / f  # Calculate periods (s) from frequencies (Hz)
    disp_curve: np.ndarray = np.zeros((t.size))
        
    solver_class = PhaseDispersion if veltype == "phase" else GroupDispersion
    forward_solver = solver_class(*velocity_model, dc=DC_VALUE)
    
    try:
        velocities = forward_solver(np.sort(t), mode=num_mode, wave=wave).velocity
    except DispersionError:
        disp_curve = np.zeros_like(f)
        return disp_curve

    if not len(velocities):
        disp_curve = np.zeros_like(f)
    else:
        disp_curve[-len(velocities):, ] = velocities[::-1] * 1000
    return disp_curve