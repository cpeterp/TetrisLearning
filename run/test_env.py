"""Basic script to test out the environment, verify rewards, actions, 
observations, etc. Input actions are passed via commandline input."""

import json

import numpy as np
from stable_baselines3.common import env_checker
from stable_baselines3.common.vec_env import DummyVecEnv

import common as cm
import memory_lookup as ml
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

    env_checker.check_env(TetrisGymEnv(env_config))
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
        print(np.reshape(obs["tilemap"][0, :], (-1, 10)))
        # print(info)
        print("drop_flag", obs["drop_flag"][0, 0])
        print("height", obs["height"][0, 0])
        print("next_shape", ml.SHAPE_LOOKUP.get(obs["next_shape"][0], None))
        print("reward", reward[0])
        print("done", done[0])
        print("step", i)
