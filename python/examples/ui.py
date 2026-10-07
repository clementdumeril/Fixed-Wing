"""Open-loop demo flight in the rl-tools ui-server (counterpart of l2f/examples/ui_min.py).

1. run `ui-server` and open http://localhost:13337
2. run this script
"""
import asyncio
import json

import numpy as np
import websockets

from fixed_wing.airplane.airplane import AILERON, ELEVATOR, THROTTLE
from fixed_wing.airplane.operations import normalize_action, step
from fixed_wing.airplane.parameters import aerosonde
from fixed_wing.airplane.trim import trim
from fixed_wing.airplane.ui import set_parameters_message, set_state_action_message, set_ui_message

AIRSPEED = 25.0
ALTITUDE = 100.0
EPISODE_SECONDS = 40.0
WIND = np.array([0.0, 4.0, 0.0])
GUST = np.array([0.0, 0.0, 6.0])  # updraft
GUST_START = 375.0  # m along x, reached after about 15 s at 25 m/s
GUST_LENGTH = 50.0  # m, about 2 s to cross


def scripted_action(trimmed_actuators, t):
    # commanded actuator positions (rad, throttle in [0, 1]) around the trim
    action = trimmed_actuators.copy()
    if 4 <= t < 5:  # aileron doublet
        action[AILERON] += np.deg2rad(4)
    elif 5 <= t < 6:
        action[AILERON] -= np.deg2rad(4)
    if 10 <= t < 11:  # elevator pulse, excites the phugoid
        action[ELEVATOR] -= np.deg2rad(4)
    if t >= 22:  # engine off, glide
        action[THROTTLE] = 0.0
    return action


async def main():
    parameters = aerosonde()
    parameters.disturbances.wind = WIND
    parameters.disturbances.gust = GUST
    parameters.disturbances.gust_start = GUST_START
    parameters.disturbances.gust_length = GUST_LENGTH
    dt = parameters.integration.dt
    async with websockets.connect("ws://localhost:13337/backend") as websocket:
        handshake = json.loads(await websocket.recv())
        assert handshake["channel"] == "handshake"
        namespace = handshake["data"]["namespace"]
        await websocket.send(set_ui_message(namespace))
        while True:
            await websocket.send(set_parameters_message(parameters, namespace))
            state = trim(parameters, AIRSPEED)
            state.position[2] = ALTITUDE
            trimmed_actuators = state.actuators.copy()
            t = 0.0
            while t < EPISODE_SECONDS and state.position[2] > 0:
                action = normalize_action(parameters, scripted_action(trimmed_actuators, t))
                state = step(parameters, state, action)
                t += dt
                await websocket.send(set_state_action_message(parameters, state, action, namespace))
                await asyncio.sleep(dt)


if __name__ == "__main__":
    asyncio.run(main())
