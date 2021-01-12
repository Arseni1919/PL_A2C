from CONSTANTS import *


class ALGNet(nn.Module):
    """
    obs_size: observation/state size of the environment
    n_actions: number of discrete actions available in the environment
    # hidden_size: size of hidden layers
    """
    def __init__(self, obs_size: int, n_action: int):
        super(ALGNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            # nn.Linear(128, 128),
            # nn.ReLU(),
            nn.Linear(128, n_action),
        )

    def forward(self, x):
        return self.net(x.float())
