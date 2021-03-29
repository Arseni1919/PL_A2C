from CONSTANTS import *
from alg_net import ALGNet


class ALGLightningModule(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.env = gym.make(ENV)
        self.state = self.env.reset()
        self.obs_size = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.n
        self.log_for_loss = []
        self.net = ALGNet(self.obs_size, self.n_actions)

        # self.agent = Agent()
        # self.total_reward = 0
        # self.episode_reward = 0

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        # rewards, log_probs, states, actions, values, Qval
        rewards = torch.cat(batch[0]).numpy()
        log_probs = batch[1]
        states = torch.cat(batch[2])
        actions = torch.cat(batch[3])
        values = batch[4]
        Qval = batch[5]

        # compute Q values
        Qvals = self.compute_Qvals(values, rewards, Qval)

        # update actor critic
        values = torch.FloatTensor(values)
        Qvals = torch.FloatTensor(Qvals)
        log_probs = torch.stack(log_probs)

        advantage = Qvals - values
        actor_loss = (-log_probs * advantage).mean()

        values, _ = self.net(states.numpy())
        values = torch.FloatTensor(values.squeeze())
        advantage = Qvals - values
        # critic_loss = F.mse_loss(values, Qvals)
        critic_loss = 0.5 * advantage.pow(2).mean()
        ac_loss = actor_loss + critic_loss + 0.001 * self.net.entropy_term

        # tr = np.sum(rewards)
        # self.log('total_reward', tr)
        tl = ac_loss.item()
        # self.log('train loss', tl, on_step=True)
        self.log_for_loss.append(tl)

        return ac_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.net.parameters(), lr=LR)

    @staticmethod
    def compute_Qvals(values, rewards, Qval):
        Qvals = np.zeros_like(values)
        for t in reversed(range(len(rewards))):
            Qval = rewards[t] + GAMMA * Qval
            Qvals[t] = Qval
        return Qvals


