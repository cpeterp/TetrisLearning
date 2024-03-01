from datetime import datetime
import json
import multiprocessing
import pytz

from stable_baselines3 import A2C
from stable_baselines3.common import env_checker
from stable_baselines3.common.vec_env import SubprocVecEnv

import common as cm
from tetris_gym import TetrisGymEnv, make_env
from tilemap_policy import TilemapCNN

if __name__ == "__main__":
    curr_time = datetime.now(pytz.timezone(cm.LOCAL_TZ))
    time_str = curr_time.strftime("%Y%m%d_%H%M")
    session_id = f"{time_str}_model"

    config_path = cm.CONFIG_DIR / "env_default.json"
    with open(config_path, "rt") as F:
        env_config = json.load(F)
        F.close()

    # Set number of CPUs dynamically
    num_cpu = multiprocessing.cpu_count()
    # Adjust number of episode incase we need free CPUs
    num_episodes = num_cpu + env_config["cpu_episode_count_mod"]

    env_checker.check_env(TetrisGymEnv(env_config))
    env = SubprocVecEnv([make_env(i, env_config) for i in range(num_cpu)])

    model = A2C(
        "MultiInputPolicy",
        env,
        verbose=1,
        n_steps=30,
        gamma=0.999999,
        # policy_kwargs={"features_extractor_class": TilemapCNN},
    )
    model.learn(total_timesteps=500000, progress_bar=True)
    print("Done Learning")
    model.save(cm.TRAINING_SESSION_DIR / session_id)
