from CONSTANTS import *


class ALGNet(nn.Module):
    """
    obs_size: observation/state size of the environment
    n_actions: number of discrete actions available in the environment
    # hidden_size: size of hidden layers
    """

    def __init__(self, obs_size: int, n_actions: int):
        super(ALGNet, self).__init__()
        hidden_size = 256
        self.critic_linear1 = nn.Linear(obs_size, hidden_size)
        self.critic_linear2 = nn.Linear(hidden_size, 1)

        self.actor_linear1 = nn.Linear(obs_size, hidden_size)
        self.actor_linear2 = nn.Linear(hidden_size, n_actions)
        self.n_actions = n_actions
        self.obs_size = obs_size
        self.entropy_term = 0

    def forward(self, state):
        state = Variable(torch.from_numpy(state).float().unsqueeze(0))
        value = F.relu(self.critic_linear1(state))
        value = self.critic_linear2(value)

        policy_dist = F.relu(self.actor_linear1(state))
        policy_dist = F.softmax(self.actor_linear2(policy_dist), dim=1)

        return value, policy_dist
