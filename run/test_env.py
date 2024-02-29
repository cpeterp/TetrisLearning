import json

from stable_baselines3.common.vec_env import DummyVecEnv

import common as cm
from tetris_gym import TetrisGymEnv, make_env


if __name__ == "__main__":
    config_path = cm.CONFIG_DIR / "env_default.json"
    with open(config_path, "rt") as F:
        env_config = json.load(F)
        F.close()

    env = DummyVecEnv([make_env(0, env_config)])
    obs = env.reset()
    for i in range(1000):
        inpt = input(
            "Choose Action: [0] Left, [1] Right, [2] Rot Left [3] Rot Right [4] None: "
        )
        action = [int(inpt)]
        obs, reward, done, info = env.step(action)
        print(obs[0, :, :, 0])
        print(info)
        print("reward", reward[0])
        print("done", done[0])
        print("step", i)
