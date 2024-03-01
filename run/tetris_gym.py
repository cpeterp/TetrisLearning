import io
from pathlib import Path
import random
from typing import Any, SupportsFloat, Tuple, Dict, List, Union

from gymnasium import Env, spaces
from gymnasium.core import RenderFrame
import numpy as np
from numpy.typing import NDArray
from pyboy import PyBoy, WindowEvent
from pyboy.botsupport.tilemap import TileMap

import common as cm
import tile_lookup as tl
import memory_lookup as ml


class TetrisGymEnv(Env):
    def __init__(self, config: dict):
        """Main Gymnasium environment for running Tetris. The config options
        allow us to use this for training and for playing back runs in a human-
        readable format"""
        # Starting values - will change during episode
        self.score = 0
        self.lines_cleared = 0
        self.shape_count = 0
        self.filled_tiles = 0
        self.shape_was_active = True
        self.is_gameover = False
        self.is_max_score = False
        self.current_tilemap = None
        self.current_rgb_array = None

        # Pulled from config
        self.visual_playback_speed = config.get("visual_playback_speed")
        self.max_shape_limit = config.get("max_shape_limit")
        self.actions_per_second = config.get("actions_per_second")
        self.run_headless = config.get("run_headless")
        self.reward_per_score = config.get("reward_per_score")
        self.reward_per_line = config.get("reward_per_line")
        self.reward_per_filled_tiles = config.get("reward_per_filled_tiles")
        self.reward_negetive_endgame = config.get("reward_negetive_endgame")
        self.reward_positive_endgame = config.get("reward_positive_endgame")
        self.end_level = config.get("end_level")
        self.end_score = config.get("end_score")
        self.end_height = config.get("end_height")
        # Which limit to use for a positive end: "level" or "score"
        self.positive_endgame = config.get("positive_endgame")
        # Ditto above: either "gameover" or "height"
        self.negetive_endgame = config.get("negetive_endgame")
        # "rgb_array"
        self.render_mode = config.get("render_mode")
        # "rgb_array" "tilemap", or dict (multi)
        self.observation_mode = config.get("observation_mode", "tilemap")
        # If true, uses the end_height+1 as the observation height
        self.truncate_play_area = config.get("truncate_play_area", False)

        # Calculate max reward
        training_level = 0  # Level currently being trained
        max_score = (
            self.reward_per_score * 1200 * 2.5 * (training_level + 1)
            + self.reward_per_line * 10 * (training_level + 1)
            + self.reward_per_filled_tiles * 10 * 10 * (training_level + 1)
            + self.reward_positive_endgame
        )

        # Super attributes
        self.reward_range = (self.reward_negetive_endgame, max_score)
        self.metadata = {
            "render_modes": ["rgb_array"],
            "render_fps": self.actions_per_second,
        }

        # Game constants
        self.max_score = 999999  # Memory won't hold a higher score
        self.max_tm_size = 32

        # Set up spaces
        self.valid_actions = [
            # WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
            None,
        ]
        self.valid_releases = [
            # WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B,
            None,
        ]
        self.action_space = spaces.Discrete(len(self.valid_actions))

        # Subset the tile map for observation
        self.tm_x_start = 2
        self.tm_x_end = 12
        self.tm_y_start = 0
        self.tm_y_end = 18
        if self.truncate_play_area:
            self.truncate_amount = self.tm_y_end - self.end_height
        else:
            self.truncate_amount = 0

        # Setup the observation space
        tm_output_shape = (
            self.tm_y_end - (self.tm_y_start + self.truncate_amount),
            self.tm_x_end - self.tm_x_start,
            1,
        )
        tm_observation_space = spaces.Box(
            low=tl.OBS_SPACE_RANGE[0],
            high=tl.OBS_SPACE_RANGE[1],
            shape=tm_output_shape,
            dtype=np.uint8,
        )
        ra_output_shape = (
            160,
            144,
            3,
        )
        ra_observation_space = spaces.Box(
            low=0,
            high=255,
            shape=ra_output_shape,
            dtype=np.uint8,
        )
        print(f"Using observation_mode {self.observation_mode}")
        if self.observation_mode == "tilemap":
            self.observation_space = tm_observation_space
        elif self.observation_mode == "rgb_array":
            self.observation_space = ra_observation_space
        elif self.observation_mode == "multi":
            self.observation_space = spaces.Dict(
                {
                    "tilemap": tm_observation_space,
                    "height": spaces.Box(0, 17, dtype=int),
                    "drop_flag": spaces.Box(0, 1, int),
                }
            )
        else:
            raise ValueError(
                f"Observation mode {self.observation_mode} is not valid."
            )

        self.hold_frames = cm.FPS // self.actions_per_second
        if self.hold_frames < 2:
            raise ValueError(
                "Provided actions_per_second is too high, must be equal/less than 2x FPS."
            )
        self.starting_state_paths = self._get_starting_state_paths()

        window_type = "headless" if self.run_headless else "SDL2"

        self.pyboy = PyBoy(
            str(cm.TETRIS_ROM_PATH),
            window_type=window_type,
            debugging=False,
            disable_input=True,
        )
        self.botsupport_manager = self.pyboy.botsupport_manager()
        # TODO: Only define this if we need to
        self.screen = self.botsupport_manager.screen()

        self.pyboy.set_emulation_speed(
            0 if self.run_headless else self.visual_playback_speed
        )

    def reset(
        self,
        seed: Union[int, None] = None,
        options: Union[Dict[str, Any], None] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        # Call super and set seed
        super().reset(seed=seed)
        self.seed = seed

        # Use seed to select a random starting state from the directory
        starting_state_i = self.np_random.integers(
            0, len(self.starting_state_paths)
        )

        starting_state_path = self.starting_state_paths[starting_state_i]
        with open(starting_state_path, "rb") as B:
            self.pyboy.load_state(B)
            B.close()

        # Set default values
        self.score = 0
        self.lines_cleared = 0
        self.shape_count = 0
        self.filled_tiles = 0
        self.shape_was_active = True
        self.is_gameover = False
        self.is_max_score = False
        self.current_tilemap = None
        self.current_rgb_array = None

        # TODO: Need to set up recording here to replay attempts

        # Grab the initial observation
        observation = self._get_observation()
        additional_info = {}  # TBD
        return observation, additional_info

    def step(
        self, action: Any
    ) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        """Advances the agent forward one step."""
        # Input comand and step pyboy forward
        self._run_action(action)

        # Update observation data
        observation = self._get_observation()

        terminated, reason = self._is_terminated()

        truncated = self._is_truncated()

        # Determine reward
        reward = self._get_reward(termination_reason=reason)

        # Obtain additional information
        additional_info = self._get_info()

        return observation, reward, terminated, truncated, additional_info

    def render(self) -> Union[RenderFrame, List[RenderFrame], None]:
        """Renders the screen as an RGB numpy array or None.

        While the Env class supports multiple render modes, this implementation
        only supports none or rgb_array"""
        if self.render_mode == "rgb_array":
            game_pixels_render = self.screen.screen_ndarray()  # (144, 160, 3)
            return game_pixels_render
        elif self.render_mode is None:
            return None
        else:
            raise ValueError(
                f"Can not use render_mode '{self.render_mode}': it is not "
                "supported. Use None or 'rgb_array'"
            )

    def _run_action(self, action: int):
        """Runs the specified action from the list of acceptable actions."""
        button_press = self.valid_actions[action]
        button_release = self.valid_releases[action]
        # One of the valid actions is None, only send non-None event
        if button_press:
            self.pyboy.send_input(button_press)
        # One tick must occur between press and release. More than 23 ticks will
        # be interpretted as a hold, which we want to avoid to keep inputs discrete
        self.pyboy.tick()
        if button_release:
            self.pyboy.send_input(button_release)
        for _ in range(0, self.hold_frames - 1):
            self.pyboy.tick()
        return None

    def close(self):
        self.pyboy.send_input(WindowEvent.QUIT)
        return None

    def _get_observation(self) -> NDArray[np.int_]:
        """Returns a simplified version of the game screen as a 20x18 NDArray.

        To generate the array, it reads in the background tilemap, filters tile
        IDs to reduce dimensionality, and overlays the sprite tiles. This
        results in the simplest view of the game screen without losing
        information loss."""
        # TODO: Consider adding a memory
        # TODO: Consider adding a timer for how many actions until drop, or any timer
        self.current_tilemap = self._get_tilemap_obs()
        if self.observation_mode == "tilemap":
            return self.current_tilemap
        elif self.observation_mode == "rgb_array":
            self.current_rgb_array = self._get_rgb_array_obs()
            return self.current_rgb_array
        elif self.observation_mode == "multi":
            obs_dict = {
                "tilemap": self.current_tilemap,
                "height": np.array([self._get_pile_height()], dtype=int),
                "drop_flag": np.array([self._shape_will_drop()], dtype=int),
            }
            return obs_dict
        else:
            raise ValueError(
                f"Observation mode {self.observation_mode} is not valid. Could not generate an observation"
            )

    def _get_rgb_array_obs(self):
        return self.screen.screen_ndarray()

    def _get_tilemap_obs(self):
        tile_size_px = 8
        # Subset tilemap to play area
        bg_tm = self.botsupport_manager.tilemap_background()[:, :]
        bg_tm = np.array(bg_tm)

        # Filter tiles to blank, filled, or wall
        conditions = [
            (bg_tm == tl.TILEMAP_BLANK_TILE),
            (
                (bg_tm >= tl.TILEMAP_SHAPE_TILES[0])
                & (bg_tm <= tl.TILEMAP_SHAPE_TILES[-1])
            ),
        ]
        values = [tl.OBS_BLANK_TILE, tl.OBS_SHAPE_TILE]
        tilemap = np.select(conditions, values, default=tl.OBS_WALL_TILE)
        # Replace background tile values with sprite tile values
        for i in range(0, 40):
            sp = self.botsupport_manager.sprite(i)
            if sp.on_screen:
                sp_x = sp.x // tile_size_px
                sp_y = sp.y // tile_size_px
                tilemap[sp_y, sp_x] = tl.OBS_SPRITE_TILE
        tilemap = tilemap.astype(np.uint8, copy=False)
        # Subset observation to play area
        # newaxis makes this a 3D array (expected for images)
        tilemap = tilemap[
            self.tm_y_start + self.truncate_amount : self.tm_y_end,
            self.tm_x_start : self.tm_x_end,
            np.newaxis,
        ]
        return tilemap

    def _is_terminated(self) -> Tuple[bool, Union[str, None]]:
        """Determines if the episode is over by checking the screen state,
        score, and level. The game episode is terminated if we get a gameover
        screen, score limit, or level limit"""
        screen_state = self.pyboy.get_memory_value(ml.SCREEN_STATE_ADDR)
        if (
            screen_state == ml.GAMEOVER_SCREEN_STATE
            or screen_state == ml.GAMEOVER_ANIMATION_SCREEN_STATE
        ):
            # Game over screen
            return True, "gameover"
        if self.positive_endgame == "score":
            if self._get_current_score() >= self.end_score:
                return True, "end_score"
        if self.positive_endgame == "level":
            if self._get_level() >= self.end_level:
                return True, "end_level"
        if self.negetive_endgame == "height":
            if self._get_pile_height() >= self.negetive_endgame:
                return True, "end_height"
        return False, None

    def _get_reward(
        self,
        termination_reason: Union[str, None] = None,
    ):
        # Calculate score from score
        new_score = self._get_current_score()
        score_diff = new_score - self.score
        self.score = new_score

        # Calculate score form cleared line
        lines_cleared = self._get_lines_cleared()
        lines_diff = lines_cleared - self.lines_cleared
        self.lines_cleared = lines_cleared

        # Calculate score from observation
        # New tiles in the last line receive score
        last_line = self.current_tilemap[-1, :]
        filled_tiles = (
            (last_line == tl.OBS_SHAPE_TILE) | (last_line == tl.OBS_SHAPE_TILE)
        ).sum()
        # Only grant positive reward for filled tiles
        if filled_tiles > self.filled_tiles:
            filled_lines_diff = filled_tiles - self.filled_tiles
        else:
            filled_lines_diff = 0
        self.filled_tiles = filled_tiles

        # Calculate score from endstates
        endgame_reward_lookup = {
            "gameover": self.reward_negetive_endgame,
            "end_height": self.reward_negetive_endgame,
            "end_score": self.reward_positive_endgame,
            "end_level": self.reward_positive_endgame,
        }
        endgame_reward = endgame_reward_lookup.get(termination_reason, 0)

        # Total reward per round
        reward = (
            score_diff * self.reward_per_score
            + lines_diff * self.reward_per_line
            + filled_lines_diff * self.reward_per_filled_tiles
            + endgame_reward
        )
        return reward

    def _is_truncated(self):
        """Checks if the episode should be truncated due to a neutral limit,
        i.e. shape count or max sore."""
        self.shape_count = self._get_shape_count()
        if self.shape_count >= self.max_shape_limit:
            return True
        elif self.score >= self.max_score:
            return True
        else:
            return False

    def _get_info(self) -> Dict:
        """Returns information about the game that is not required for observations."""
        additional_info = {
            "level": self._get_level(),
            "score": self.score,
            "lines_cleared": self.lines_cleared,
            "shape_count": self.shape_count,
            "filled_tiles": self.filled_tiles,
            "shape_was_active": self.shape_was_active,
            "is_gameover": self.is_gameover,
            "is_max_score": self.is_max_score,
            "will_drop": self._shape_will_drop(),
            "pile_height": self._get_pile_height(),
        }
        additional_info.update(self._get_shapes())

        return additional_info

    def _get_starting_state_paths(self) -> List[io.BytesIO]:
        """Returns starting state paths that can be read by PyBoy"""
        starting_states = []
        for fp in cm.POST_START_STATE_DIR.glob("*.state"):
            starting_states.append(fp)
        return starting_states

    def _get_current_score(self) -> int:
        """Returns the current score of the game.

        Tetris records its score as a 3 byte little endian BCD starting at
        SCORE_ADDR_0. Each byte stores a two digit hex value corresponding to
        the digits of a decimal number. E.g. 0x55, 0x03, 0x01 -> 10355
        """
        score_hex_values = [
            self.pyboy.get_memory_value(ml.SCORE_ADDR_0),
            self.pyboy.get_memory_value(ml.SCORE_ADDR_1),
            self.pyboy.get_memory_value(ml.SCORE_ADDR_2),
        ]
        # Each byte stores 2 digits as decimals
        score_decimal_value = (
            score_hex_values[0] % 16
            + (score_hex_values[0] // 16) * 10
            + (score_hex_values[1] % 16) * 100
            + (score_hex_values[1] // 16) * 1000
            + (score_hex_values[2] % 16) * 10000
            + (score_hex_values[2] // 16) * 100000
        )
        return score_decimal_value

    def _get_shapes(self) -> Dict:
        """Returns the active, preview, and next preview shapes with rotation"""
        active_shape = ml.SHAPE_LOOKUP.get(
            self.pyboy.get_memory_value(ml.ACTIVE_SHAPE_ADDR), None
        )
        preview_shape = ml.SHAPE_LOOKUP.get(
            self.pyboy.get_memory_value(ml.PREVIEW_SHAPE_ADDR), None
        )
        next_preview_shape = ml.SHAPE_LOOKUP.get(
            self.pyboy.get_memory_value(ml.NEXT_PREVIEW_SHAPE_ADDR), None
        )
        shape_dict = {
            "active_shape": active_shape,
            "preview_shape": preview_shape,
            "next_preview_shape": next_preview_shape,
        }
        return shape_dict

    def _get_lines_cleared(self) -> int:
        """The number of lines cleared is also stared as a 3 byte little endian
        BCD"""
        lines_hex_values = [
            self.pyboy.get_memory_value(ml.LINES_CLEARED_ADDR_0),
            self.pyboy.get_memory_value(ml.LINES_CLEARED_ADDR_1),
            self.pyboy.get_memory_value(ml.LINES_CLEARED_ADDR_2),
        ]
        # Each byte stores 2 digits as decimals
        lines_decimal_value = (
            lines_hex_values[0] % 16
            + (lines_hex_values[0] // 16) * 10
            + (lines_hex_values[1] % 16) * 100
            + (lines_hex_values[1] // 16) * 1000
            + (lines_hex_values[2] % 16) * 10000
            + (lines_hex_values[2] // 16) * 100000
        )
        return lines_decimal_value

    def _get_shape_count(self) -> int:
        """Returns the current count of shapes played.

        Tetris exposes a flag in memory indicating when a shape is active (in
        play) and when it is inactive (e.g. immediately after a shape drops, or
        while a line clears). By checking when this status chages, we can
        increment the count of shapes."""
        shape_status = self.pyboy.get_memory_value(ml.ACTIVE_SHAPE_FLAG_ADDR)
        shape_is_active = shape_status == ml.ACTIVE_SHAPE_FLAG

        if shape_is_active and not self.shape_was_active:
            # i.e. status switched form inactive to active - indicating a new
            # shape coming
            new_shape_count = self.shape_count + 1
        else:
            new_shape_count = self.shape_count
        self.shape_was_active = shape_is_active
        return new_shape_count

    def _get_level(self) -> int:
        """Returns the current Tetris level"""
        level_hex = self.pyboy.get_memory_value(ml.GAME_LEVEL_ADDR)
        level_decimal = level_hex % 16 + (level_hex // 16) * 10
        return level_decimal

    def _shape_will_drop(self) -> bool:
        """Returns an indicator whether the shape will drop the next step or
        not. Drops occur the frame after the timer is equal to the delay.

        E.g. if the value of the delay 28, the shape drops a level between timer
        values 28 and 27."""
        # If shape is has dropped all the way, return none
        shape_pending_state = self.pyboy.get_memory_value(ml.DROP_STATE_ADRR)
        if shape_pending_state in (ml.SHAPE_DROPPED, ml.NEXT_SHAPE_PENDING):
            return False
        delay = self.pyboy.get_memory_value(ml.DROP_DELEAY_ADDR)
        timer = self.pyboy.get_memory_value(ml.DROP_TIMER_ADDR)

        # Returns true if the shape will be lower after the next step
        if ((timer + 1) % (delay + 1)) - self.hold_frames <= 0:
            return True
        return False

    def _get_pile_height(self) -> int:
        """Gets the distance from the ceiling to the top of the pile. The pile
        has a minimum height of o and a max height of 17 (the last tile can't be
        filled)."""
        if self.current_tilemap is None:
            tilemap_obs = self._get_tilemap_obs()
        else:
            tilemap_obs = self.current_tilemap
        pile_coords = np.where(tilemap_obs == tl.OBS_SHAPE_TILE)
        if len(pile_coords[0]) < 1:
            return 0
        # Row index 0 corresponds to tile w/ height 18
        height = 18 - pile_coords[0][0]
        return height

    # Scratch
    # If ml.NEXT_SHAPE_PENDING, ignore the timer value.


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
