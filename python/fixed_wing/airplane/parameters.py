"""Aircraft parameter sets, one function per aircraft (counterpart of l2f/parameters/dynamics/*.h)"""
import numpy as np

from .airplane import Aerodynamics, Disturbances, Dynamics, Integration, Parameters

SIMULATION_FREQUENCY = 100

# body (x forward, y right, z down) <-> body (x forward, y left, z up)
FRD_FLU = np.diag([1.0, -1.0, -1.0])


def aerosonde():
    # Placeholder aircraft until the ARPL parameters are available.
    # Airframe and coefficients: Aerosonde UAV, Beard & McLain, "Small Unmanned Aircraft", appendix E.
    # thrust_max and the actuator time constants are not from the book, they are guesses.
    Jx, Jy, Jz, Jxz = 0.8244, 1.135, 1.759, 0.1204
    J_frd = np.array([
        [Jx, 0, -Jxz],
        [0, Jy, 0],
        [-Jxz, 0, Jz],
    ])
    J = FRD_FLU @ J_frd @ FRD_FLU
    surface_limit = np.deg2rad(30)
    return Parameters(
        dynamics=Dynamics(
            mass=13.5,
            gravity=np.array([0, 0, -9.81]),
            J=J,
            J_inv=np.linalg.inv(J),
            air_density=1.2682,
            wing_area=0.55,
            wing_span=2.8956,
            mean_chord=0.18994,
            aerodynamics=Aerodynamics(
                C_L_0=0.28, C_L_alpha=3.45, C_L_q=0.0, C_L_delta_e=-0.36,
                C_D_p=0.0437, oswald_efficiency=0.9,
                C_Y_0=0.0, C_Y_beta=-0.98, C_Y_p=0.0, C_Y_r=0.0, C_Y_delta_a=0.0, C_Y_delta_r=-0.17,
                C_l_0=0.0, C_l_beta=-0.12, C_l_p=-0.26, C_l_r=0.14, C_l_delta_a=0.08, C_l_delta_r=0.105,
                C_m_0=-0.02338, C_m_alpha=-0.38, C_m_q=-3.6, C_m_delta_e=-0.5,
                C_n_0=0.0, C_n_beta=0.25, C_n_p=0.022, C_n_r=-0.35, C_n_delta_a=0.06, C_n_delta_r=-0.032,
            ),
            thrust_max=40.0,
            actuator_time_constants=np.array([0.05, 0.05, 0.05, 0.2]),
            actuator_limits=np.array([
                [-surface_limit, surface_limit],
                [-surface_limit, surface_limit],
                [-surface_limit, surface_limit],
                [0.0, 1.0],
            ]),
        ),
        integration=Integration(dt=1.0 / SIMULATION_FREQUENCY),
        disturbances=Disturbances(wind=np.zeros(3)),
    )
