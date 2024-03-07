from datetime import datetime
import json
import multiprocessing
import pytz

from stable_baselines3 import A2C, PPO
from stable_baselines3.common import env_checker
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

import common as cm
from save_best_model_callback import SaveOnBestTrainingRewardCallback
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
    env = VecMonitor(
        SubprocVecEnv([make_env(i, env_config) for i in range(num_cpu)]),
        str(cm.LOG_DIR),
    )

    # Set learning vars
    gamma = 0.99
    learning_rate = 0.0003
    n_steps = 2048

    # model = PPO(
    #     "MultiInputPolicy",
    #     env,
    #     verbose=1,
    #     n_steps=n_steps,
    #     gamma=gamma,
    #     learning_rate=learning_rate,
    #     ent_coef=0.1,
    #     tensorboard_log=cm.TENSORBOARD_LOG_DIR,
    #     policy_kwargs={"features_extractor_kwargs": {"cnn_output_dim": 512}},
    # )

    model = A2C.load(cm.BEST_MODEL_PATH, env=env)

    callback = SaveOnBestTrainingRewardCallback(
        check_freq=100,
        log_dir=cm.LOG_DIR,
    )

    tb_log_name = (
        f"PPO_{str(gamma)[2:]}_{str(learning_rate)[2:]}_{str(n_steps)}"
    )
    model.learn(
        total_timesteps=5_000_000,
        progress_bar=True,
        callback=callback,
        tb_log_name=tb_log_name,
    )
    print("Done Learning")
    model.save(cm.TRAINING_SESSION_DIR / session_id)
