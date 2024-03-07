## Reinforcement learning on GB Tetris

1) Obtain a ROM of Tetris for Gameboy, move to the `lib` directory, and rename `tetris.gb`
2) In your commandline, navigate to the TetrisLearning directory and install conda environment with with `conda env create -p ./.conda environment.yml`. You may swap out the `-p ./.conda` argument with `-n some_name` if you prefer.
3) Activate the environment: `conda activate ./.conda`
4) Train the model by running `python run/run_training.py`
5) [Optional] Track progress of the model training via Tensorboard. While the model is running, open a seperate terminal, navigate to the TetrisLearning directory, and run `tensorboard --logdir ./board`
6) Once training has finished, play the model using the `play_model.ipynb` notbook

### Resources
- [Pan Docs](https://bgb.bircd.org/pandocs.htm): Low-level details on the Gameboy
- [Reverse Engineering the Gameboy Tetrs](https://github.com/h3nnn4n/Reverse-Engineering-the-GameBoy-Tetris/): Memory locations in GB Tetris
- [RAM Map for GB Tetris](https://datacrystal.romhacking.net/wiki/Tetris_(Game_Boy)/RAM_map)
- [Paper: Human-level control through deep reinforcement learning](https://www.readcube.com/articles/10.1038%2Fnature14236?shared_access_token=Lo_2hFdW4MuqEcF3CVBZm9RgN0jAjWel9jnR3ZoTv0P5kedCCNjz3FJ2FhQCgXkApOr3ZSsJAldp-tw3IWgTseRnLpAc9xQq-vTA2Z5Ji9lg16_WvCy4SaOgpK5XXA6ecqo8d8J7l4EJsdjwai53GqKt-7JuioG0r3iV67MQIro74l6IxvmcVNKBgOwiMGi8U0izJStLpmQp6Vmi_8Lw_A%3D%3D): The setup for the CNN we use, as well as the frame stacking method are pretty much directly pulled from this paper. See the methodology section for specifics. For a simpler overview, see this [related presentation from the authors](https://courses.engr.illinois.edu/cs546/sp2018/Slides/Apr05_Minh.pdf)
- [Paper: Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/pdf/1602.01783.pdf): The A2C model used is from this paper.
