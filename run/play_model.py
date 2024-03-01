import json
import multiprocessing
from pathlib import Path
import time

import numpy as np
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

import common as cm
import memory_lookup as ml
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
        print(np.reshape(obs["tilemap"][0, :], (-1, 10)))
        print("height", obs["height"][0, 0])
        print("next_shape", ml.SHAPE_LOOKUP.get(obs["next_shape"][0], None))
        # print(info)
        print(reward)

# TODO: normalize observation, add layer to policy, flatten tilemap obs, monitor model, then try truncating scene
