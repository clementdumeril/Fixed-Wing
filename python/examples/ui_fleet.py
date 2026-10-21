"""Many airplanes at once in the rl-tools ui-server, each on its own test scenario (like Isaac Lab's parallel
environments): the PID, then the policy at several stages of training. Airplanes whose flight ended turn red.

1. run `ui-server` and open http://localhost:13337
2. python examples/ui_fleet.py runs/sac_123_seed0 --n 36 --stages 100000 500000 1000000

The flights are simulated once and cached in <run>/fleet_cache_<n>.npz, then replayed in a loop at 2x real time.
"""
import argparse
import asyncio
import json
import re
from pathlib import Path

import numpy as np
import websockets

from fixed_wing.airplane.environment import EnvConfig, FixedWingEnv
from fixed_wing.airplane.pid import PIDPolicy
from fixed_wing.airplane.ui import set_parameters_message, set_ui_message

SAMPLE = 0.1  # s between two frames
SPEED = 2.0  # replay speed, times real time


def record(env, policy, seed):
    observation, _ = env.reset(seed=seed)
    policy.reset(env)
    every = round(SAMPLE * env.config.frequency)
    positions, orientations = [env.state.position.copy()], [env.state.orientation.copy()]
    done = False
    while not done:
        observation, _, term, trunc, _ = env.step(policy(env, observation))
        done = term or trunc
        if env.steps % every == 0 or done:
            positions.append(env.state.position.copy())
            orientations.append(env.state.orientation.copy())
    return np.array(positions), np.array(orientations), bool(term)


def simulate(run, n, stages):
    """Returns [(title, color, positions (n, T, 3), orientations (n, T, 4), last sample (n,), ended (n,))]."""
    from stable_baselines3 import SAC

    from fixed_wing.airplane.asymmetric_sac import SB3Policy

    checkpoints = {int(re.search(r"(\d+)_steps", p.name).group(1)): p for p in (run / "checkpoints").glob("*_steps.zip")}
    pilots = [("PID", "#2a6fdb", PIDPolicy())]
    pilots += [(f"RL after {s:,} training steps", "#e8743b", SB3Policy(SAC.load(checkpoints[s], device="cpu"))) for s in stages]
    env = FixedWingEnv(EnvConfig(privileged_observation=True))
    T = 1 + round(env.config.episode_seconds / SAMPLE)
    fleets = []
    for title, color, policy in pilots:
        positions, orientations = np.zeros((n, T, 3)), np.zeros((n, T, 4))
        last, ended = np.zeros(n, dtype=int), np.zeros(n, dtype=bool)
        for i in range(n):
            p, q, ended[i] = record(env, policy, i)
            last[i] = len(p) - 1
            positions[i, :len(p)], orientations[i, :len(p)] = p, q
            positions[i, len(p):], orientations[i, len(p):] = p[-1], q[-1]  # stays where it ended
        print(f"{title}: {ended.sum()} of {n} flights ended early")
        fleets.append((title, color, positions, orientations, last, ended))
    return fleets, env.parameters


async def play(fleets, parameters):
    async with websockets.connect("ws://localhost:13337/backend") as websocket:
        namespace = json.loads(await websocket.recv())["data"]["namespace"]
        await websocket.send(set_ui_message(namespace, "ui_fleet.js"))
        while True:
            for title, color, positions, orientations, last, ended in fleets:
                await websocket.send(set_parameters_message(parameters, namespace))
                for k in range(positions.shape[1]):
                    fleet = [dict(position=positions[i, k].tolist(), orientation=orientations[i, k].tolist(),
                                  ended=bool(ended[i] and k >= last[i])) for i in range(len(positions))]
                    await websocket.send(json.dumps({"namespace": namespace, "channel": "setState", "data": {
                        "state": dict(fleet=fleet, title=title, color=int(color[1:], 16), t=k * SAMPLE), "action": []}}))
                    await asyncio.sleep(SAMPLE / SPEED)
                await asyncio.sleep(2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path, help="directory written by examples/train_sac.py")
    parser.add_argument("--n", type=int, default=36, help="airplanes, test scenarios 0 to n-1")
    parser.add_argument("--stages", type=int, nargs="+", default=[100_000, 500_000, 1_000_000])
    args = parser.parse_args()
    cache = args.run / f"fleet_cache_{args.n}_{'_'.join(map(str, args.stages))}.npz"
    if cache.exists():
        data = np.load(cache, allow_pickle=True)
        fleets, parameters = list(data["fleets"]), data["parameters"].item()
    else:
        fleets, parameters = simulate(args.run, args.n, args.stages)
        np.savez(cache, fleets=np.array(fleets, dtype=object), parameters=np.array(parameters, dtype=object))
    asyncio.run(play(fleets, parameters))


if __name__ == "__main__":
    main()
