"""PID baseline for the environment task, the classic successive loop closure (Beard & McLain, chapter 6).

course -> roll -> aileron, altitude -> pitch -> elevator, airspeed -> throttle; rudder at zero.
It only uses what the airplane measures (environment.measurements), and the trim of the nominal airplane as
feedforward, so it gets no more information than the RL actor.
"""
from dataclasses import dataclass

import numpy as np

from .airplane import AILERON, ELEVATOR, THROTTLE, Parameters
from .environment import Setpoint, measurements, wrap
from .operations import normalize_action
from .parameters import aerosonde
from .trim import trim


@dataclass
class PIDGains:
    course_p: float = 1.5  # rad of roll per rad of course error
    roll_max: float = np.deg2rad(30)
    roll_p: float = 0.8  # rad of aileron per rad of roll error
    roll_d: float = 0.1  # rad of aileron per rad/s of roll rate
    altitude_p: float = 0.04  # rad of pitch per m
    altitude_i: float = 0.003  # rad of pitch per m.s
    pitch_max: float = np.deg2rad(15)  # around the trim pitch
    pitch_p: float = 0.6  # rad of elevator per rad of pitch error
    pitch_d: float = 0.5  # rad of elevator per rad/s of pitch rate
    airspeed_p: float = 0.15  # throttle per m/s
    airspeed_i: float = 0.02  # throttle per m


class PIDController:
    def __init__(self, dt, nominal: Parameters = None, gains: PIDGains = None):
        self.dt = dt
        self.nominal = nominal or aerosonde()
        self.gains = gains or PIDGains()

    def reset(self, setpoint: Setpoint):
        self.setpoint = setpoint
        trimmed = trim(self.nominal, setpoint.airspeed)
        self.trim_actuators = trimmed.actuators.copy()
        self.trim_pitch = -2 * np.arctan2(trimmed.orientation[2], trimmed.orientation[0])  # nose-up rotation about -y
        self.altitude_integral = 0.0
        self.airspeed_integral = 0.0

    def __call__(self, m) -> np.ndarray:
        """Normalized action from the measurements of environment.measurements."""
        g, s = self.gains, self.setpoint
        p, q_body, _ = m["rates"]
        nose_up_rate = -q_body  # body y points left

        roll_command = np.clip(-g.course_p * wrap(s.course - m["course"]), -g.roll_max, g.roll_max)  # left turn: roll < 0
        aileron = g.roll_p * (roll_command - m["roll"]) - g.roll_d * p

        altitude_error = s.altitude - m["altitude"]
        self.altitude_integral = np.clip(self.altitude_integral + altitude_error * self.dt, -50, 50)
        pitch_offset = np.clip(g.altitude_p * altitude_error + g.altitude_i * self.altitude_integral, -g.pitch_max, g.pitch_max)
        elevator = self.trim_actuators[ELEVATOR] - g.pitch_p * (self.trim_pitch + pitch_offset - m["pitch"]) + g.pitch_d * nose_up_rate

        airspeed_error = s.airspeed - m["airspeed"]
        self.airspeed_integral = np.clip(self.airspeed_integral + airspeed_error * self.dt, -20, 20)
        throttle = self.trim_actuators[THROTTLE] + g.airspeed_p * airspeed_error + g.airspeed_i * self.airspeed_integral

        command = np.zeros(4)
        command[AILERON], command[ELEVATOR], command[THROTTLE] = aileron, elevator, throttle
        return np.clip(normalize_action(self.nominal, command), -1, 1)


class PIDPolicy:
    """PIDController with the interface of environment.evaluate."""

    def __init__(self, gains: PIDGains = None):
        self.gains = gains

    def reset(self, env):
        self.controller = PIDController(1 / env.config.frequency, env.nominal, self.gains)
        self.controller.reset(env.setpoint)

    def __call__(self, env, observation):
        return self.controller(measurements(env.parameters, env.state, env.config))
