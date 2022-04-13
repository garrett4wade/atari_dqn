from collections import deque
import argparse
import gym
import logging
import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from core import ReplayBuffer, AtariDQN, DQN
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
            terminal_on_life_loss=True,
        )
        env = gym.wrappers.FrameStack(env, 4)
        if eval_:
            # env = gym.wrappers.RecordVideo(env, './results')
            pass
        else:
            env = gym.wrappers.TransformReward(env, lambda x: np.sign(x))
    return env


def eval_dqn(
        agent,
        eval_env,
        n_episodes=20,
        device=T.device("cuda:0")
    if T.cuda.is_available() else T.device("cpu"),
):
    ep_cnt = total_ep_step = total_ep_ret = 0
    obs = eval_env.reset()
    while ep_cnt < n_episodes:
        action = agent.choose_action(obs)
        obs, _, done, info = eval_env.step(action)
        for d, inf in zip(done, info):
            if d:
                ep_cnt += 1
                total_ep_ret += inf['episode']['r']
                total_ep_step += inf['episode']['l']
    return dict(
        eval_episode_return=total_ep_ret / n_episodes,
        eval_episode_length=total_ep_step / n_episodes,
    )


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

        if len(self.input_dims) == 1:
            self.q_eval = DQN(
                *self.input_dims,
                self.n_actions,
                dueling=dueling,
                linear=linear,
            )
        else:
            self.q_eval = AtariDQN(
                self.n_actions,
                linear=linear,
                dueling=dueling,
            )

        self.optimizer = T.optim.Adam(self.q_eval.parameters(), lr=lr)

        if len(self.input_dims) == 1:
            self.q_next = DQN(
                *self.input_dims,
                self.n_actions,
                dueling=dueling,
                linear=linear,
            )
        else:
            self.q_next = AtariDQN(
                self.n_actions,
                linear=linear,
                dueling=dueling,
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
        return loss.item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="CartPole-v1")
    parser.add_argument("--group", type=str)
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
    eval_interval = int(1e3)
    log_interval = int(1e3)
    save_interval = int(5e4)

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

    scores = deque(maxlen=100)
    losses = deque(maxlen=1000)

    n_env_steps = ep_cnt = 0
    observation = train_env.reset()
    while n_env_steps < total_env_steps:
        action = agent.choose_action(observation)
        observation_, reward, done, info = train_env.step(action)

        agent.memory.put_batch(observation, action, observation_, reward, done)
        loss = agent.learn()

        if loss is not None:
            losses.append(loss)

        observation = observation_

        for d, inf in zip(done, info):
            if d:
                ep_cnt += 1
                scores.append(inf['episode']['r'])

        n_env_steps += 1

        if n_env_steps % log_interval == 0:
            avg_score = np.mean(scores)
            logger.info(''.join([
                f'Environment Steps {n_env_steps}/{total_env_steps}',
                f'\t Number of Episodes: {ep_cnt}',
                '\t Average Score %.1f' % avg_score,
                '\t Epsilon %.2f' % agent.epsilon
            ]))
            if wandb_run is not None and args.wandb:
                wandb.log(
                    dict(train_episode_return=avg_score,
                         eps=agent.epsilon,
                         loss=np.mean(losses)), n_env_steps)

        if n_env_steps % eval_interval == 0:
            eval_info = eval_dqn(agent, eval_env)
            logger.info(
                f"Evaluation Episode Return {eval_info['eval_episode_return']}"
                f", Episode Length {eval_info['eval_episode_length']}.")
            if wandb_run is not None and args.wandb:
                wandb.log(eval_info, n_env_steps)

        if n_env_steps % save_interval == 0:
            fname = "dqn"
            if args.linear:
                fname += "_linear"
            if args.double:
                fname += "_double"
            T.save(agent.q_eval.state_dict(),
                   f"results/{fname}_{n_env_steps}.pt")

    if wandb_run is not None:
        wandb_run.finish()