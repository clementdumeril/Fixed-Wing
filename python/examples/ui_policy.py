"""A trained policy (or the PID) flying one test scenario in the rl-tools ui-server, in real time.

1. run `ui-server` and open http://localhost:13337
2. python examples/ui_policy.py --model runs/sac_123_seed0/model.zip --seed 0     (without --model: the PID)
"""
import argparse
import asyncio
import json

import websockets

from fixed_wing.airplane.environment import EnvConfig, FixedWingEnv
from fixed_wing.airplane.pid import PIDPolicy
from fixed_wing.airplane.ui import set_parameters_message, set_state_action_message, set_ui_message


async def main(args):
    env = FixedWingEnv(EnvConfig(privileged_observation=True))
    if args.model:
        from stable_baselines3 import SAC

        from fixed_wing.airplane.asymmetric_sac import SB3Policy

        policy = SB3Policy(SAC.load(args.model, device="cpu"))
    else:
        policy = PIDPolicy()
    dt = 1 / env.config.frequency
    async with websockets.connect("ws://localhost:13337/backend") as websocket:
        handshake = json.loads(await websocket.recv())
        namespace = handshake["data"]["namespace"]
        await websocket.send(set_ui_message(namespace))
        while True:  # replays the scenario until interrupted
            observation, _ = env.reset(seed=args.seed)
            policy.reset(env)
            await websocket.send(set_parameters_message(env.parameters, namespace))
            done = False
            while not done:
                action = policy(env, observation)
                observation, _, term, trunc, info = env.step(action)
                done = term or trunc
                await websocket.send(set_state_action_message(env.parameters, env.state, action, namespace))
                await asyncio.sleep(dt)
            print(f"scenario {args.seed}: {info['termination'] or 'flew the whole episode'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="model.zip of examples/train_sac.py")
    parser.add_argument("--seed", type=int, default=0, help="test scenario")
    asyncio.run(main(parser.parse_args()))
