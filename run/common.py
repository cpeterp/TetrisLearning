from pathlib import Path

# System settings
LOCAL_TZ = "US/Eastern"  # Change to your local tz, used for file naming

# Paths
TETRIS_ROM_PATH = Path.cwd() / "data/tetris.gb"
POST_START_STATE_DIR = Path.cwd() / "data/variable_starting_states"
STARTING_STATE_PATH = Path.cwd() / "data/default/game_a_start_menu.state"
TRAINING_SESSION_DIR = Path.cwd() / "data/sessions"
CONFIG_DIR = Path.cwd() / "config"

# Game data
FPS = 60
