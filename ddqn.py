import argparse
import gym
import logging
import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from core import ReplayBuffer
from env_wrapper import SubprocVecEnv

logging.basicConfig(
    format=
    "%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")

logger = logging.getLogger('atari')
logger.setLevel(logging.INFO)

fh = logging.FileHandler('results/log.txt', mode='w')
fh.setLevel(logging.INFO)
logger.addHandler(fh)

try:
    import wandb
except ModuleNotFoundError:
    logger.info(
        "wandb not installed. Code runs fine without logging to the web.")


def pixel_to_float(x):
    if x.dtype == np.uint8 and len(x.shape) > 1:
        return np.array(x, dtype=np.float32) / 255
    else:
        return x


class DQN(nn.Module):

    def __init__(self, obs_dim, act_dim, linear, dueling, hidden_dim=512):
        super().__init__()

        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.linear = linear
        self.dueling = dueling
        if dueling:
            self.V = nn.Linear(hidden_dim, 1)
            self.A = nn.Linear(hidden_dim, act_dim)
        else:
            self.Q = nn.Linear(hidden_dim, act_dim)

        self.device = T.device("cpu")

    def forward(self, state):
        act_fn = (lambda x: x) if self.linear else F.relu
        flat1 = act_fn(self.fc1(state))
        if self.dueling:
            V = self.V(flat1)
            A = self.A(flat1)

            return V + A - A.mean(-1, keepdim=True)
        else:
            return self.Q(flat1)


def make_env(env_name, eval_=False):
    env = gym.make(env_name)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    if "NoFrameskip" in env_name:
        env = gym.wrappers.AtariPreprocessing(
            env,
            noop_max=30,
            frame_skip=4,
            screen_size=84,
            scale_obs=False,
        )
        env = gym.wrappers.FrameStack(env, 4)
        if eval_:
            # env = gym.wrappers.RecordVideo(env, './results')
            pass
    return env


def eval_dqn(q_net,
             eval_env,
             n_episodes=20,
             device=T.device("cuda:0"),
             deterministic=False,
             eps=0.05):
    ep_cnt = total_ep_step = total_ep_ret = total_time = 0
    while ep_cnt < n_episodes:
        obs = eval_env.reset()
        done = False
        while not done:
            obs_net = T.from_numpy(pixel_to_float(obs)).unsqueeze(0).to(device)
            act = q_net(obs_net).argmax(-1).item()
            if not deterministic and np.random.rand() < eps:
                act = eval_env.action_space.sample()
            obs, _, done, info = eval_env.step(act)
        ep_cnt += 1
        total_ep_ret += info['episode']['r']
        total_ep_step += info['episode']['l']
        total_time += info['episode']['t']
    logger.info(
        f"Evaluation {n_episodes}, Episode Return {total_ep_ret / n_episodes}"
        f", Episode Length {total_ep_step / n_episodes}"
        f", Time/Episode {total_time / n_episodes}s ({total_time}s in total)")
    return dict(eval_episode_return=total_ep_ret / n_episodes,
                eval_episode_length=total_ep_step / n_episodes,
                eval_time=total_time / n_episodes)


class Agent():

    def __init__(self,
                 gamma,
                 epsilon,
                 lr,
                 n_actions,
                 input_dims,
                 mem_size,
                 batch_size,
                 double,
                 linear,
                 dueling,
                 eps_min=0.01,
                 eps_dec=5e-7,
                 replace=1000,
                 chkpt_dir='tmp/dueling_ddqn'):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.replace_target_cnt = replace
        self.chkpt_dir = chkpt_dir
        self.action_space = [i for i in range(self.n_actions)]
        self.learn_step_counter = 0

        self.double = double

        self.memory = ReplayBuffer(mem_size, input_dims)

        self.q_eval = DQN(
            *self.input_dims,
            self.n_actions,
            dueling=dueling,
            linear=linear,
        )
        self.optimizer = T.optim.Adam(self.q_eval.parameters(), lr=lr)

        self.q_next = DQN(
            *self.input_dims,
            self.n_actions,
            dueling=dueling,
            linear=linear,
        )

    def choose_action(self, observation):
        state = T.tensor(pixel_to_float(observation),
                         dtype=T.float).to(self.q_eval.device)
        q = self.q_eval.forward(state)
        dtm_action = T.argmax(q, -1).cpu().numpy()
        rnd_action = np.array([
            np.random.choice(self.action_space)
            for _ in range(observation.shape[0])
        ])
        mask = np.random.random(size=(observation.shape[0], )) > self.epsilon

        return mask * dtm_action + (1 - mask) * rnd_action

    def learn(self):
        if self.memory.cur_size < self.batch_size:
            return

        self.optimizer.zero_grad()

        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

        state, action, new_state, reward, done = \
                                self.memory.get(self.batch_size)

        states = T.tensor(pixel_to_float(state)).to(self.q_eval.device)
        rewards = T.tensor(reward).to(self.q_eval.device)
        dones = T.tensor(done).to(self.q_eval.device)
        actions = T.tensor(action).to(self.q_eval.device)
        states_ = T.tensor(pixel_to_float(new_state)).to(self.q_eval.device)

        indices = np.arange(self.batch_size)

        q_pred = self.q_eval.forward(states)[indices, actions]
        q_next = self.q_next.forward(states_)

        q_eval = self.q_eval.forward(states_)

        max_actions = T.argmax(q_eval, dim=1)

        q_next[dones] = 0.0
        if self.double:
            q_next = q_next[indices, max_actions]
        else:
            q_next = q_next.max(-1).values
        q_target = rewards + self.gamma * q_next

        loss = ((q_target - q_pred)**2).mean()
        loss.backward()
        self.optimizer.step()
        self.learn_step_counter += 1

        self.epsilon = max(self.eps_min, self.epsilon - self.eps_dec)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="CartPole-v1")
    parser.add_argument("--dueling", action='store_true')
    parser.add_argument("--double", action="store_true")
    parser.add_argument("--linear", action='store_true')
    parser.add_argument("--wandb", action='store_true')
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    T.backends.cudnn.deterministic = True
    T.backends.cudnn.benchmark = True

    np.random.seed(args.seed)
    T.manual_seed(args.seed)
    T.cuda.manual_seed_all(args.seed)

    try:
        import wandb
        if args.wandb:
            wandb_run = wandb.init(project='atari_dqn',
                                   config=vars(args),
                                   group=args.group,
                                   name=f"seed{args.seed}")
        else:
            wandb_run = None
    except ModuleNotFoundError:
        wandb_run = None
        logger.info(
            "wandb not installed. Code runs fine without logging to the web.")

    device = T.device("cuda:0") if T.cuda.is_available() else T.device('cpu')

    train_env = SubprocVecEnv(
        [lambda: make_env(args.env_name) for _ in range(8)])
    eval_env = SubprocVecEnv(
        [lambda: make_env(args.env_name, eval_=True) for _ in range(2)])

    act_dim = train_env.action_space.n

    total_env_steps = int(5e6)

    agent = Agent(gamma=0.99,
                  epsilon=1.0,
                  lr=5e-4,
                  input_dims=train_env.observation_space.shape,
                  n_actions=train_env.action_space.n,
                  mem_size=int(1e5),
                  dueling=args.dueling,
                  linear=args.linear,
                  double=args.double,
                  eps_min=0.01,
                  batch_size=64,
                  eps_dec=1e-3,
                  replace=100)
    scores = []

    n_env_steps = i = 0
    observation = train_env.reset()
    while n_env_steps < total_env_steps:
        action = agent.choose_action(observation)
        observation_, reward, done, info = train_env.step(action)

        agent.memory.put_batch(observation, action, observation_, reward, done)
        agent.learn()

        observation = observation_

        for d, inf in zip(done, info):
            if d:
                i += 1
                scores.append(inf['episode']['r'])
                avg_score = np.mean(scores[max(0, i - 100):(i + 1)])
                print('episode: ', i, 'score %.1f ' % inf['episode']['r'],
                      ' average score %.1f' % avg_score,
                      'epsilon %.2f' % agent.epsilon)

    x = [i + 1 for i in range(num_games)]

    if wandb_run is not None:
        wandb_run.finish()