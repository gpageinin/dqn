from random import random
from time import sleep

import gym
import tensorboardX as tbx
import torch
import torch.nn.functional as F
import torch.optim as opt
from tqdm import tqdm

from agent import Agent


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ハイパーパラメータ
HIDDEN_NUM = 32  # エージェントの隠れ層のニューロン数
EPISODE_NUM = 2000  # 何エピソード実行するか
GAMMA = .99  # 時間割引率

agent = Agent(HIDDEN_NUM).to(device)
env = gym.make('CartPole-v0')
optimizer = opt.Adam(agent.parameters())


# 状態を受け取り、行動価値最大の行動とその行動価値を返す
def calc_q(obs, train=False, random_act=False):
    # PyTorchで使える形式に変換
    obs = torch.tensor(obs, dtype=torch.float, device=device).unsqueeze(0)

    # 行動価値を計算
    if train:
        agent.train()
        qs = agent(obs).squeeze()
    else:
        agent.eval()
        with torch.no_grad():
            qs = agent(obs).squeeze()

    # 行動を決定
    if random_act:  # ランダム行動
        action = env.action_space.sample()
    else:  # 行動価値最大の行動
        action = torch.argmax(qs).item()

    return action, qs[action]


# 1エピソード（`done`が`True`になるまで）行動し続ける
# 1回行動する度にエージェントを更新する
def do_episode(epsilon, render=False):
    obs = env.reset()
    done = False
    reward_sum = 0

    while not done:
        # 描画（学習後に結果を確認する用）
        if render:
            env.render()
            sleep(.01)

        # 確率epsilonでランダム行動
        action, q = calc_q(obs, train=True, random_act=(random() < epsilon))
        next_obs, reward, done, _ = env.step(action)

        reward_sum += reward

        # 誤差の計算に用いる次の状態でのQ値の計算
        if done:  # `done`が`True`なら次の状態は存在しないため、そのQ値は0
            next_q = torch.zeros((), dtype=torch.float32, device=device)
        else:
            _, next_q = calc_q(next_obs)  # max_{a} Q(s_{t+1}, a)

        loss = F.mse_loss(q, reward + GAMMA*next_q)

        # エージェントを更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        obs = next_obs

    env.close()

    return reward_sum


if __name__ == '__main__':
    with tbx.SummaryWriter() as writer:
        try:
            for episode in tqdm(range(1, EPISODE_NUM + 1)):
                # epsilonを線形に小さくする
                epsilon = .5 - episode * .5 / EPISODE_NUM
                reward_sum = do_episode(epsilon)

                # 1エピソードで得られた報酬和を記録
                writer.add_scalar('data/reward_sum', reward_sum, episode)
        finally:
            env.close()
            # 学習結果を描画する
            do_episode(epsilon=0, render=True)

