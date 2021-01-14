from CONSTANTS import *
from alg_lightning_module import ALGLightningModule
from alg_net import ALGNet


def get_action(state, net):
    value, policy_dist = net(state)
    _, action = torch.max(policy_dist, dim=1)
    return int(action.item())


def play(times: int = 1):
    env = gym.make(ENV)
    state = env.reset()

    model = ALGNet(env.observation_space.shape[0], env.action_space.n)
    model.load_state_dict(torch.load("example.ckpt"))

    game = 0
    total_reward = 0
    while game < times:
        # action = env.action_space.sample()
        action = get_action(state, model)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        env.render()
        if done:
            state = env.reset()
            game += 1
            print(f'finished game {game} with a total reward: {total_reward}')
            total_reward = 0
        else:
            state = next_state
    env.close()


if __name__ == '__main__':
    play(10)
