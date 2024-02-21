import io
from pathlib import Path
import random
from typing import Any, SupportsFloat, Tuple, Dict, List

from gymnasium import Env, spaces
from gymnasium.core import RenderFrame
import numpy as np
from pyboy import PyBoy, WindowEvent
from pyboy.botsupport.tilemap import TileMap

import common
import tile_lookup as tl
import memory_lookup as ml


class TetrisGymEnv(Env):
    def __init__(self, config: dict):
        """Main Gymnasium environment for running Tetris. The config options
        allow us to use this for training and for playing back runs in a human-
        readable format"""
        self.score = 0
        self.lines_cleared = 0
        self.level = 0
        self.top_line = 0
        self.current_shape = None
        self.next_shape = None

        run_headless = config["run_headless"]

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

        self.starting_states = self.load_starting_states()

        self.output_shape = (18, 20, 3)
        self.frame_stacks = 2
        self.col_steps = 16
        self.output_full = (
            # Stacking consecutive input frames permits memory for the agent
            self.output_shape[0] * self.frame_stacks,
            self.output_shape[1],
            self.output_shape[2],
        )

        # Set these in ALL subclasses
        self.action_space = spaces.Discrete(len(self.valid_actions))
        self.observation_space = spaces.Box(
            low=0, high=255, shape=self.output_full, dtype=np.uint8
        )

        window_type = "headless" if run_headless else "SDL2"

        self.pyboy = PyBoy(
            str(common.TETRIS_ROM_PATH),
            window_type=window_type,
            debugging=False,
            disable_input=True,
        )
        self.botsupport_manager = self.pyboy.botsupport_manager()
        self.pyboy.set_emulation_speed(
            0 if run_headless else config["visual_playback_speed"]
        )

    def reset(
        self, seed: int | None = None, options: Dict[str, Any] | None = None
    ) -> Tuple[Any | Dict[str, Any]]:
        # Use seed to select a random starting state from the directory
        self.seed = seed
        random(seed)
        starting_state_i = random.randint(0, len(self.starting_states) - 1)
        starting_state = self.starting_states[starting_state_i]
        self.pyboy.load_state(starting_state)

        # Set default values
        self.score = 0
        self.lines_cleared = 0
        self.level = 0
        self.top_line = 0
        # TODO: Define functions to pull these from memory
        self.current_shape = None
        self.next_shape = None

        # TODO: Need to set up recording here to replay attempts

        # Grab the initial observation
        observation = self.get_observation()
        additional_info = {}  # TBD
        return observation, additional_info

    def step(
        self, action: Any
    ) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        """Advances the agent forward one step."""
        self.run_action(action)

        observation = self.get_observation()

        # Determine reward
        reward = self.get_reward()

        # Check if terminated
        terminated = self.is_terminated()

        # Obtain additional information
        additional_info = self.get_info()

        return observation, reward, terminated, truncated, additional_info

    def render(self) -> RenderFrame | List[RenderFrame] | None:
        return super().render()

    def run_action(self, action):
        return None

    def load_starting_states(self) -> List[io.BytesIO]:
        """Loads all starting states as BytesIO that can be read by PyBoy"""
        starting_states = []
        for fp in common.POST_START_STATE_DIR.glob("*.state"):
            with open(fp, "rb") as B:
                starting_state_bytes = B.read()
                B.close()
            starting_states.append(io.BytesIO(starting_state_bytes))
        return starting_states

    def get_observation(self) -> np.ndarray[int]:
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

    def get_reward(self):
        new_score = self.get_current_score()
        if new_score >= self.score:
            score_diff = new_score - self.score
        else:
            # The bits rolled over
            # TODO: Test + issue warning/note/log if this happens
            score_diff = (999999 - self.score) + new_score
        self.score = new_score

        lines_cleared = self.get_lines_cleared()
        if lines_cleared >= self.lines_cleared:
            lines_diff = lines_cleared - self.lines_cleared
        else:
            # The bits rolled over
            # TODO: Test + issue warning/note/log if this happens
            lines_diff = (999999 - self.lines_cleared) + lines_cleared
        self.lines_cleared = lines_cleared

        reward = (
            score_diff * common.REWARD_MULT_SCORE
            + lines_diff * common.REWARD_MULT_LINES
        )
        return reward

    def get_current_score(self) -> int:
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

    def get_info(self) -> Dict:
        """Returns information about the game that is not required for observations."""
        additional_info = {}
        additional_info.update(self.get_shapes)
        return additional_info

    def get_shapes(self) -> Dict:
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

    def get_lines_cleared(self) -> Dict[str, int]:
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

    def is_terminated(self) -> bool:
        """Determines if the game is over by checking the screen state"""
        screen_state = self.pyboy.get_memory_value(ml.SCREEN_STATE_ADDR)
        if (
            screen_state == ml.GAMEOVER_SCREEN_STATE
            or screen_state == ml.GAMEOVER_ANIMATION_SCREEN_STATE
        ):
            return True
        return False


if __name__ == "__main__":
    config = {"run_headless": True, "visual_playback_speed": 5}
    tge = TetrisGymEnv(config)
