"""Messages for the rl-tools ui-server (pip install ui-server), same protocol and names as the l2f Python bindings.

The server relays these JSON messages to the browser, which runs the render module in ui.js.
"""
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .airplane import Parameters, State
from .operations import air_data, specific_energy, wind


def _json(value):
    if isinstance(value, dict):
        return {k: _json(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def get_ui() -> str:
    return (Path(__file__).parent / "ui.js").read_text(encoding="utf-8")


def set_ui_message(namespace: str) -> str:
    return json.dumps({
        "namespace": namespace,
        "channel": "setUI",
        "latch": True,
        "data": {"type": "2d", "render_function": get_ui()},
    })


def set_parameters_message(parameters: Parameters, namespace: str) -> str:
    return json.dumps({
        "namespace": namespace,
        "channel": "setParameters",
        "latch": True,  # so that a browser opened after the start of the episode gets them
        "data": _json(asdict(parameters)),
    })


def set_state_action_message(parameters: Parameters, state: State, action, namespace: str) -> str:
    _, airspeed, alpha, beta = air_data(parameters, state)
    state_json = _json(asdict(state))
    state_json["air_data"] = {"airspeed": float(airspeed), "alpha": float(alpha), "beta": float(beta)}
    state_json["specific_energy"] = float(specific_energy(parameters, state))
    state_json["wind"] = _json(wind(parameters, state.position))  # local wind, gust included
    return json.dumps({
        "namespace": namespace,
        "channel": "setState",
        "data": {"state": state_json, "action": _json(np.asarray(action))},
    })
