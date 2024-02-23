import io
from pathlib import Path
import random
from typing import Any, SupportsFloat, Tuple, Dict, List

from gymnasium import Env, spaces
from gymnasium.core import RenderFrame
import numpy as np
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
        self._shape_was_active = True
        self._is_gameover = False
        self._is_max_score = False

        # Pulled from config
        self.visual_playback_speed = config["visual_playback_speed"]
        self.max_shape_limit = config["max_shape_limit"]
        self.run_headless = config["run_headless"]
        self.reward_per_score = config["reward_per_score"]
        self.reward_per_line = config["reward_per_line"]
        self.reward_gameover = config["reward_gameover"]
        self.reward_max_score = config["reward_max_score"]
        self.actions_per_second = config["actions_per_second"]

        # Constants
        self.max_score = 999999
        self.valid_actions = [
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
        ]
        self.valid_releases = [
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B,
        ]
        self.hold_frames = 60 // self.actions_per_second
        self.starting_states = self._load_starting_states()

        self.output_shape = (18, 20, 1)

        # Set these in ALL subclasses
        self.action_space = spaces.Discrete(len(self.valid_actions))
        self.observation_space = spaces.Box(
            low=0, high=2, shape=self.output_shape, dtype=np.uint8
        )

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
        self, seed: int | None = None, options: Dict[str, Any] | None = None
    ) -> Tuple[Any | Dict[str, Any]]:
        # Call super and set seed
        super().reset(seed=seed)
        self.seed = seed

        # Use seed to select a random starting state from the directory
        starting_state_i = self.np_random.integers(0, len(self.starting_states))
        starting_state = self.starting_states[starting_state_i]
        self.pyboy.load_state(starting_state)

        # Set default values
        self.score = 0
        self.lines_cleared = 0
        self.shape_count = 0
        self._shape_was_active = True

        # TODO: Need to set up recording here to replay attempts

        # Grab the initial observation
        observation = self._get_observation()
        additional_info = {}  # TBD
        return observation, additional_info

    def step(
        self, action: Any
    ) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        """Advances the agent forward one step."""
        self.run_action(action)

        observation = self._get_observation()

        terminated, reason = self._is_terminated()

        truncated = self._is_truncated()

        # Determine reward
        reward = self._get_reward(termination_reason=reason)

        # Obtain additional information
        additional_info = self._get_info()

        return observation, reward, terminated, truncated, additional_info

    def render(self) -> RenderFrame | List[RenderFrame] | None:
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

    def run_action(self, action: int):
        self.pyboy.send_input(self.valid_actions[action])
        for _ in range(0, self.hold_frames):
            self.pyboy.tick()
        self.pyboy.send_input(self.valid_releases[action])
        self.pyboy.tick()
        return None

    def close(self):
        self.pyboy.send_input(WindowEvent.QUIT)
        return None

    def _get_observation(self) -> np.ndarray[int]:
        """Returns a simplified version of the game screen as a 20x18 NDArray.

        To generate the array, it reads in the background tilemap, filters tile
        IDs to reduce dimensionality, and overlays the sprite tiles. This
        results in the simplest view of the game screen without losing
        information loss."""
        bg_tm = self.botsupport_manager.tilemap_background()[0:20, 0:18]
        bg_tm = np.array(bg_tm)

        conditions = [
            (bg_tm == tl.TILEMAP_BLANK_TILE),
            (bg_tm >= tl.TILEMAP_SHAPE_TILES[0])
            & (bg_tm <= tl.TILEMAP_SHAPE_TILES[-1]),
        ]
        values = [tl.OBS_BLANK_TILE, tl.OBS_SHAPE_TILE]
        obs_tm = np.select(conditions, values, default=tl.OBS_WALL_TILE)

        for i in range(0, 40):
            sp = self.botsupport_manager.sprite(i)
            if sp.on_screen:
                sp_x = sp.x // 8
                sp_y = sp.y // 8
                # All onscreen sprites are shape tiles
                obs_tm[sp_y, sp_x] = tl.OBS_SHAPE_TILE
        return obs_tm

    def _is_terminated(self) -> Tuple[bool, str | None]:
        """Determines if the episode is over by checking the screen state and
        score. The game episode is terminated if we get a gameover screen or
        the score is over 999,999 (max score that can be measured)"""
        screen_state = self.pyboy.get_memory_value(ml.SCREEN_STATE_ADDR)
        if (
            screen_state == ml.GAMEOVER_SCREEN_STATE
            or screen_state == ml.GAMEOVER_ANIMATION_SCREEN_STATE
        ):
            # Game over screen
            return True, "gameover"
        if self.score >= self.max_score:
            return True, "max_score"
        return False, None

    def _get_reward(self, termination_reason: str | None = None):
        new_score = self._get_current_score()
        score_diff = new_score - self.score
        self.score = new_score

        lines_cleared = self._get_lines_cleared()
        lines_diff = lines_cleared - self.lines_cleared
        self.lines_cleared = lines_cleared

        gameover_reward = (
            self.reward_gameover if termination_reason == "gameover" else 0
        )
        max_score_reward = (
            self.reward_max_score if termination_reason == "max_score" else 0
        )

        reward = (
            score_diff * self.reward_per_score
            + lines_diff * self.reward_per_line
            + gameover_reward
            + max_score_reward
        )
        return reward

    def _is_truncated(self):
        """Checks if the episode should be truncated by comparing the current
        count of shapes to the limit provided in the config"""
        self.shape_count = self._get_shape_count()
        if self.shape_count >= self.max_shape_limit:
            return True
        else:
            return False

    def _get_info(self) -> Dict:
        """Returns information about the game that is not required for observations."""
        additional_info = {}
        additional_info.update(self._get_shapes())
        additional_info.update(self._get_level())
        return additional_info

    def _load_starting_states(self) -> List[io.BytesIO]:
        """Loads all starting states as BytesIO that can be read by PyBoy"""
        starting_states = []
        for fp in cm.POST_START_STATE_DIR.glob("*.state"):
            with open(fp, "rb") as B:
                starting_state_bytes = B.read()
                B.close()
            starting_states.append(io.BytesIO(starting_state_bytes))
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

    def _get_lines_cleared(self) -> Dict[str, int]:
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

        if shape_is_active and not self._shape_was_active:
            # i.e. status switched form inactive to active - indicating a new
            # shape coming
            new_shape_count = self.shape_count + 1
        else:
            new_shape_count = self.shape_count
        self._shape_was_active = shape_is_active
        return new_shape_count

    def _get_level(self) -> int:
        """Returns the current Tetris level"""
        level_hex = self.pyboy.get_memory_value(ml.GAME_LEVEL_ADDR)
        level_decimal = level_hex % 16 + (level_hex // 16) * 10
        return {"level": level_decimal}


if __name__ == "__main__":
    config = {"run_headless": True, "visual_playback_speed": 5}
    tge = TetrisGymEnv(config)
