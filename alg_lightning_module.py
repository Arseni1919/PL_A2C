from CONSTANTS import *
from alg_net import ALGNet


class ALGLightningModule(pl.LightningModule):

    def __init__(self, dataset):
        super().__init__()
        self.env = gym.make(ENV)
        self.state = self.env.reset()
        self.obs_size = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.n

        self.net = ALGNet(self.obs_size, self.n_actions)
        self.target_net = ALGNet(self.obs_size, self.n_actions)

        self.dataset = dataset

        # self.agent = Agent()
        self.total_reward = 0
        self.episode_reward = 0

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        epsilon = max(EPS_END, EPS_START - self.global_step / EPS_LAST_FRAME)
        # print(f'\n{self.global_step} - {EPS_LAST_FRAME} - {epsilon}\n')
        reward, done = self._play_step(epsilon)
        self.episode_reward += reward
        if done:
            self.total_reward = self.episode_reward
            self.episode_reward = 0

        loss = self._dqn_mse_loss(batch)

        if self.global_step % SYNC_RATE:
            self.target_net.load_state_dict(self.net.state_dict())

        self.log('current total reward', self.total_reward)
        self.log('train loss', loss)
        # self.log('epsilon', epsilon)
        # if batch_idx % 1000 == 0:
        #     print(epsilon)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.net.parameters(), lr=LR)

    @torch.no_grad()
    def _play_step(self, epsilon):
        action = self._get_action(epsilon)
        new_state, reward, done, _ = self.env.step(action)
        exp = Experience(self.state, action, reward, done, new_state)
        self.dataset.append(exp)
        self.state = self.env.reset() if done else new_state
        return reward, done

    def _get_action(self, epsilon) -> int:
        if np.random.random() < epsilon:
            return self.env.action_space.sample()
        else:
            state = torch.tensor([self.state])
            q_values = self.net(state)
            _, action = torch.max(q_values, dim=1)
            return int(action.item())

    def _dqn_mse_loss(self, batch):
        states, actions, rewards, dones, nex_states = batch
        state_action_values = self.net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            next_state_values = self.target_net(nex_states).max(1)[0]
            next_state_values[dones] = 0.0
            next_state_values = next_state_values.detach()

        expected_state_action_values = next_state_values * GAMMA + np.array(rewards, dtype=np.float32)

        return nn.MSELoss()(state_action_values, expected_state_action_values)
