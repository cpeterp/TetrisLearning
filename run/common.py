from pathlib import Path

# System settings
LOCAL_TZ = "US/Eastern"  # Change to your local tz, used for file naming

# Paths
TETRIS_ROM_PATH = Path.cwd() / "data/tetris.gb"
POST_START_STATE_DIR = Path.cwd() / "data/variable_starting_states"
STARTING_STATE_PATH = Path.cwd() / "data/default/game_a_start_menu.state"
TRAINING_SESSION_DIR = Path.cwd() / "data/sessions"
BEST_MODEL_PATH = Path.cwd() / "data/logs/best_model.zip"
CONFIG_DIR = Path.cwd() / "config"
LOG_DIR = Path.cwd() / "data/logs"
TENSORBOARD_LOG_DIR = Path.cwd() / "board"

# Game data
FPS = 60
