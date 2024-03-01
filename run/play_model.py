import json
import multiprocessing
from pathlib import Path
import time

from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

import common as cm
from tetris_gym import TetrisGymEnv, make_env

if __name__ == "__main__":
    # Select the latest model
    model_paths = [fp for fp in cm.TRAINING_SESSION_DIR.glob("*.zip")]
    model_paths.sort(reverse=True)
    model_path = model_paths[0]
    print(f"Loading model {model_path}")

    config_path = cm.CONFIG_DIR / "env_default.json"
    with open(config_path, "rt") as F:
        env_config = json.load(F)
        F.close()

    # DummyVecEnv lets us run the multi cpu env w/ 1 env
    env = DummyVecEnv([make_env(0, env_config)])
    model = A2C.load(model_path, env=env)

    obs = env.reset()
    for i in range(1000):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        time.sleep(0.05)
        print(obs[0, :, :, 0])
        # print(info)
        # print(reward)
