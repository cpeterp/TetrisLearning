from gymnasium import Space, spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th
from torch import nn
from torch.optim.adam import Adam as Adam
from torch.optim.optimizer import Optimizer as Optimizer


class TilemapCNN(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: Space,
        features_dim: int = 256,
        normalized_image: bool = False,
    ) -> None:
        assert isinstance(observation_space, spaces.Box), (
            "TilemapCNN must be used with a gym.spaces.Box ",
            f"observation space, not {observation_space}",
        )
        super().__init__(observation_space, features_dim)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=1, padding=0),
            nn.ReLU(),
            # nn.Conv2d(32, 64, kernel_size=2, stride=2, padding=0),  # 15x7 ->
            # nn.ReLU(),
            # nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            # nn.ReLU(),
            nn.Flatten(),
        )
        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim), nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))
