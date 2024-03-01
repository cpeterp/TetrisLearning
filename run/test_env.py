"""Basic script to test out the environment, verify rewards, actions, 
observations, etc. Input actions are passed via commandline input."""

import json

from stable_baselines3.common.vec_env import DummyVecEnv

import common as cm
from tetris_gym import TetrisGymEnv, make_env

inpt_lu = {
    "a": 0,
    "d": 1,
    "w": 2,
    "s": 3,
    " ": 4,
}

if __name__ == "__main__":
    config_path = cm.CONFIG_DIR / "env_default.json"
    with open(config_path, "rt") as F:
        env_config = json.load(F)
        F.close()

    env = DummyVecEnv([make_env(0, env_config)])
    obs = env.reset()
    for i in range(1000):
        while True:
            inpt = input(
                "Choose Action: [a] Left, [d] Right, [w] Rot Left [s] Rot Right [ ] None: "
            )
            action = inpt_lu.get(inpt, None)
            if action is not None:
                break
        obs, reward, done, info = env.step([int(action)])
        print(obs[0, :, :, 0])
        print(info)
        print("reward", reward[0])
        print("done", done[0])
        print("step", i)
