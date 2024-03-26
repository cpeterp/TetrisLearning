from datetime import datetime
import json
import multiprocessing
import pytz

from stable_baselines3 import A2C, PPO
from stable_baselines3.common import env_checker
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    VecMonitor,
    VecFrameStack,
)
from torch import nn as nn

import common as cm
from save_best_model_callback import SaveOnBestTrainingRewardCallback
from tetris_gym import TetrisGymEnv, make_env
from utils import linear_schedule


if __name__ == "__main__":
    config_path = cm.CONFIG_DIR / "env_default.json"
    with open(config_path, "rt") as F:
        env_config = json.load(F)
        F.close()

    hparams_path = cm.CONFIG_DIR / "ppo_hparams.json"
    with open(hparams_path, "rt") as F:
        hparams = json.load(F)
        F.close()

    # Set number of CPUs dynamically
    num_cpu = multiprocessing.cpu_count()

    # Create the envs
    env_checker.check_env(TetrisGymEnv(env_config))
    vec_env = SubprocVecEnv([make_env(i, env_config) for i in range(num_cpu)])
    # vec_env = VecFrameStack(vec_env, 4)
    env = VecMonitor(
        vec_env,
        str(cm.LOG_DIR),
    )

    # Set model learning hyperparameters
    net_arch = {
        "none": [],
        "tiny": dict(pi=[64], vf=[64]),
        "small": dict(pi=[64, 64], vf=[64, 64]),
        "medium": dict(pi=[256, 256], vf=[256, 256]),
    }[hparams.get("net_arch", "tiny")]

    activation_fn = {
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "elu": nn.ELU,
        "leaky_relu": nn.LeakyReLU,
    }[hparams.get("activation_fn", "relu")]

    clip_range = linear_schedule(hparams.get("clip_range_m", 0.1))
    learning_rate = linear_schedule(hparams.get("learning_rate_m", 2.5e-4))
    # clip_range = hparams.get("clip_range", 0.1)
    # learning_rate = hparams.get("learning_rate", 2.5e-4)
    batch_size = hparams.get("batch_size_agent", 32) * num_cpu

    model_kwargs = {
        "n_steps": hparams.get("n_steps", 128),
        "batch_size": hparams.get("batch_size", 256),
        "gamma": hparams.get("gamma", 0.99),
        "learning_rate": learning_rate,
        "ent_coef": hparams.get("ent_coef", 0.01),
        "clip_range": clip_range,
        "n_epochs": hparams.get("n_epochs", 3),
        "gae_lambda": hparams.get("gae_lambda", 0.95),
        "vf_coef": hparams.get("vf_coef", 0.5),
        "policy_kwargs": {
            "net_arch": net_arch,
            "activation_fn": activation_fn,
            "features_extractor_kwargs": {
                "cnn_output_dim": hparams.get("cnn_output_dim", 512)
            },
        },
    }

    [print(f"{key}: {val}") for key, val in model_kwargs.items()]

    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        tensorboard_log=cm.TENSORBOARD_LOG_DIR,
        **model_kwargs,
    )
    # model = PPO.load(cm.BEST_MODEL_PATH, env=env)

    callback = SaveOnBestTrainingRewardCallback(
        check_freq=model_kwargs["n_steps"],
        log_dir=cm.LOG_DIR,
        save_path=cm.BEST_MODEL_PATH,
    )

    # Gets the total_timesteps closest to target that is a multiple of n_steps and num_cpus
    target_total_timesteps = 500000
    total_timesteps = (
        model_kwargs["n_steps"]
        * num_cpu
        * ((target_total_timesteps // (model_kwargs["n_steps"] * num_cpu)) + 1)
    )

    for _ in range(4):
        curr_time = datetime.now(pytz.timezone(cm.LOCAL_TZ))
        time_str = curr_time.strftime("%Y%m%d_%H%M")
        session_id = f"{time_str}_model"
        tb_log_name = f"PPO_{session_id}"

        model.learn(
            total_timesteps=total_timesteps,
            log_interval=5,
            progress_bar=True,
            callback=callback,
            tb_log_name=tb_log_name,
        )
        print("Done Learning")
        model.save(cm.TRAINING_SESSION_DIR / session_id)
        model = PPO.load(cm.BEST_MODEL_PATH, env=env)
