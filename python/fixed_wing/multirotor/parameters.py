"""Port of rl_tools/rl/environments/l2f/parameters/dynamics/arpl.h and parameters/default.h"""
import numpy as np

from .multirotor import Dynamics, Integration, Parameters

SIMULATION_FREQUENCY = 100

_J = np.diag([1.7665e-3, 1.3766e-3, (1.7665e-3 + 1.3766e-3) / 2 * 1.879])


def arpl():
    return Parameters(
        dynamics=Dynamics(
            rotor_positions=np.array([
                [0.0775, -0.0981, 0],
                [-0.0775, 0.0981, 0],
                [0.0775, 0.0981, 0],
                [-0.0775, -0.0981, 0],
            ]),
            rotor_thrust_directions=np.array([[0, 0, 1.0]] * 4),
            rotor_torque_directions=np.array([[0, 0, -1.0], [0, 0, -1.0], [0, 0, 1.0], [0, 0, 1.0]]),
            rotor_thrust_coefficients=np.array([[-0.5300192490953086, 8.971435376396558, -0.8911788522090669]] * 4),
            rotor_torque_constants=np.full(4, 0.005964552),
            rotor_time_constants_rising=np.full(4, 0.033),
            rotor_time_constants_falling=np.full(4, 0.033),
            mass=0.782,
            gravity=np.array([0, 0, -9.81]),
            J=_J,
            J_inv=np.linalg.inv(_J),
            action_limit=(0.0, 1.0),
        ),
        integration=Integration(dt=1.0 / SIMULATION_FREQUENCY),
    )
