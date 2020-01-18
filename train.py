from random import random

import gym
import tensorboardX as tbx
import torch
import torch.nn.functional as F
import torch.optim as opt
from tqdm import tqdm

from agent import Agent


# ハイパーパラメータ
HIDDEN_NUM = 32  # エージェントの隠れ層のニューロン数
EPISODE_NUM = 2000  # 何エピソード実行するか
GAMMA = .99  # 時間割引率

agent = Agent(HIDDEN_NUM)
env = gym.make('CartPole-v0')
optimizer = opt.Adam(agent.parameters())


# 状態を受け取り、行動価値最大の行動とその行動価値を返す
def calc_q(obs, train=False, random_act=False):
    obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    if train:
        agent.train()
        qs = agent(obs).squeeze()
    else:
        agent.eval()
        with torch.no_grad():
            qs = agent(obs).squeeze()

    if random_act:
        action = env.action_space.sample()
    else:
        action = torch.argmax(qs).item()

    return action, qs[action]


def do_episode(epsilon):
    obs = env.reset()
    done = False
    reward_sum = 0

    while not done:
        # 確率epsilonでランダム行動
        action, q = calc_q(obs, train=True, random_act=(random() < epsilon))
        next_obs, reward, done, _ = env.step(action)

        reward_sum += reward

        # エージェントを更新
        if done:
            next_q = torch.zeros((), dtype=torch.float32)  # doneなら次の状態は存在せず行動価値も0
        else:
            _, next_q = calc_q(next_obs)  # max_{a} Q(s_{t+1}, a)
        loss = F.mse_loss(q, reward + GAMMA*next_q)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        obs = next_obs

    return reward_sum


if __name__ == '__main__':
    with tbx.SummaryWriter() as writer:
        for episode in tqdm(range(1, EPISODE_NUM + 1)):
            epsilon = .5 - episode * .5 / EPISODE_NUM  # epsilonを線形に小さくする
            reward_sum = do_episode(epsilon)
            writer.add_scalar('data/reward_sum', reward_sum, episode)

