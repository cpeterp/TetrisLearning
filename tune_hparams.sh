#!/bin/bash
./.conda/bin/python rl-baselines3-zoo/train.py \
    --optimize-hyperparameters \
    --algo ppo \
    --env TetrisGym-v0 \
    --env-kwargs config:\"./config/env_default.json\" \
    --vec-env subproc \
    --conf-file ./config/ppo_hyperparameters.yml \
    --study-name tg_ppo_opt \
    --storage sqlite:///tg_ppo_opt.db \
    --tensorboard-log ./board \
    --optimization-log-path data/logs \
    --n-timesteps 100000 \
    --n-trials 1000 \
    --n-jobs 5 \
    --progress
