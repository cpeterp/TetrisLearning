import json
import multiprocessing
from os.path import exists
from pathlib import Path
import uuid

from stable_baselines3 import A2C, PPO
from stable_baselines3.common import env_checker
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback

from tetris_gym import TetrisGymEnv
import common as cm


def make_env(rank, env_conf, seed=0):
    """
    Utility function for multiprocessed env.

    Params:
        rank: (int) index of the subprocess
        env_id: (str) the environment ID
        num_env: (int) the number of environments you wish to have in subprocesses
        seed: (int) the initial seed for RNG
    """

    def _init():
        env = TetrisGymEnv(env_conf)
        # Ensure a different see per Env
        env.reset(seed=(seed + rank))
        return env

    return _init


if __name__ == "__main__":
    # ep_length = 2048 * 8
    # sess_path = cm.TRAINING_SESSION_DIR / f"session_{str(uuid.uuid4())[:8]}"

    config_path = cm.CONFIG_DIR / "default.json"
    with open(config_path, "rt") as F:
        env_config = json.load(F)
        F.close()

    # Set number of CPUs dynamically
    num_cpu = multiprocessing.cpu_count()
    # Adjust number of episode incase we need free CPUs
    num_episodes = num_cpu + env_config["cpu_episode_count_mod"]

    frames_per_step = cm.FPS // env_config["actions_per_second"]
    frames_per_drop = 954  # 18 row x 53 frames per row at level 0
    steps_per_drop = frames_per_drop // frames_per_step
    n_steps = steps_per_drop * 3

    env = TetrisGymEnv(env_config)
    # env = SubprocVecEnv([make_env(i, env_config) for i in range(num_cpu)])
    # env = make_env(0, env_config)()

    # checkpoint_callback = CheckpointCallback(
    #     save_freq=ep_length, save_path=sess_path, name_prefix="tetris"
    # )

    env_checker.check_env(env)
    # learn_steps = 40

    model = A2C("MlpPolicy", env, verbose=1, n_steps=15)
    model.learn(total_timesteps=1000)

    vec_env = model.get_env()

    obs = vec_env.reset()
    for i in range(1000):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        print(action)
        vec_env.render("human")

    # for i in range(learn_steps):
    #     model.learn(
    #         total_timesteps=(ep_length) * num_cpu * 1000,
    #         callback=checkpoint_callback,
    #     )
