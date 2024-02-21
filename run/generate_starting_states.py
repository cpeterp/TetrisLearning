import io
from multiprocessing import Pool
from pathlib import Path
from random import randint

from pyboy import PyBoy, WindowEvent
import sys

print(sys.path)
import common


# TODO: Convert to wrapper
def generate_starting_states(n: int, remove_old: bool = True):
    """Generates n random starting states for tetris. Reads a start in the
    start menu, waits a random number of frames, and starts the game."""
    # Setup directory for post-start state files
    common.POST_START_STATE_DIR.mkdir(parents=True, exist_ok=True)

    # Read in state file to memory, rather than having each process open and
    # close the file (which could block eachother)
    with open(common.STARTING_STATE_PATH, "rb") as B:
        starting_state_bytes = B.read()
        B.close()

    # Generate arguments for each process and apply them
    pool_args = []
    for _ in range(0, n):
        pool_args.append(
            (
                randint(1, 60),  # wait_frames
                io.BytesIO(starting_state_bytes),  # starting_state_bio
            )
        )
    with Pool(n) as p:
        new_save_states = p.starmap(_gen_starting_state, pool_args)

    # Write out new save states to files
    for i, post_start_state_bytes in enumerate(new_save_states):
        # Get file path
        post_start_state_path = common.POST_START_STATE_DIR.joinpath(
            f"post_start_{i}.state"
        )
        # Delete if desired
        if post_start_state_path.exists() and remove_old:
            post_start_state_path.unlink()

        with open(post_start_state_path, "wb") as B:
            B.write(post_start_state_bytes)
            B.close()

    return None


def _gen_starting_state(wait_frames: int, starting_state_bio: io.BytesIO):
    # Instantiate PyBoy with Tetris Rom
    pb = PyBoy(
        str(common.TETRIS_ROM_PATH),
        disable_renderer=True,
        window_type="headless",
        debugging=False,
        disable_input=True,
    )
    pb.set_emulation_speed(0)

    # Load starting point (game mode A, before game begins)
    pb.load_state(starting_state_bio)

    # Wait random frames to randomize starting seed/first shapes
    for i in range(0, wait_frames):
        pb.tick()
    pb.send_input(WindowEvent.PRESS_BUTTON_START)
    pb.tick()
    pb.send_input(WindowEvent.RELEASE_BUTTON_START)

    # Save state to Bytes IO object, then convert to bytes
    post_start_state_bio = io.BytesIO()
    post_start_state_bio.seek(0)
    pb.save_state(post_start_state_bio)

    post_start_state_bio.seek(0)
    post_start_state_bytes = post_start_state_bio.read()
    return post_start_state_bytes


if __name__ == "__main__":
    generate_starting_states(10)
