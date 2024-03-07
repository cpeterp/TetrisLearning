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
from utils import linear_schedule

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

    env_checker.check_env(TetrisGymEnv(env_config))
    env = VecMonitor(
        SubprocVecEnv([make_env(i, env_config) for i in range(num_cpu)]),
        str(cm.LOG_DIR),
    )

    # Set learning vars
    gamma = 0.99
    learning_rate = 2.5e-4
    n_steps = 128
    ent_coef = 0.01
    vf_coef = 0.5
    clip_range = 0.1
    gae_lambda = 0.95
    batch_size = 256
    n_epochs = 3

    # model = PPO(
    #     "MultiInputPolicy",
    #     env,
    #     verbose=1,
    #     n_epochs=n_epochs,
    #     n_steps=n_steps,
    #     gamma=gamma,
    #     learning_rate=linear_schedule(learning_rate),
    #     vf_coef=vf_coef,
    #     ent_coef=ent_coef,
    #     clip_range=linear_schedule(clip_range),
    #     gae_lambda=gae_lambda,
    #     batch_size=batch_size,
    #     tensorboard_log=cm.TENSORBOARD_LOG_DIR,
    #     policy_kwargs={"features_extractor_kwargs": {"cnn_output_dim": 512}},
    # )

    model = PPO.load(cm.BEST_MODEL_PATH, env=env)

    callback = SaveOnBestTrainingRewardCallback(
        check_freq=100,
        log_dir=cm.LOG_DIR,
    )

    tb_log_name = (
        f"PPO_{str(gamma)[2:]}_{str(learning_rate)[2:]}_{str(n_steps)}"
    )
    model.learn(
        total_timesteps=n_steps * num_cpu * 1000,
        log_interval=5,
        progress_bar=True,
        callback=callback,
        tb_log_name=tb_log_name,
    )
    print("Done Learning")
    model.save(cm.TRAINING_SESSION_DIR / session_id)
