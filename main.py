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
from env_wrapper import SubprocVecEnv, wrap_deepmind

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



def make_env(env_name, eval_=False):
    env = gym.make(env_name)
    if "NoFrameskip" in env_name:
        env = wrap_deepmind(env, episode_life=(not eval_), clip_rewards=(not eval_), frame_stack=True, scale=True)
        env = gym.wrappers.TransformObservation(env, lambda x: np.transpose(x, (2, 0, 1)))
    return env


def eval_dqn(
        agent,
        eval_env,
        n_episodes=20,
):
    ep_cnt = 0
    ep_steps = []
    ep_rets = []
    obs = eval_env.reset()
    running_rewards = np.zeros(eval_env.nenvs, dtype=np.float32)
    running_ep_len = np.zeros(eval_env.nenvs, dtype=np.float32)
    while ep_cnt < n_episodes:
        action = agent.choose_action(obs, eps=0.05)
        obs, r, done, info = eval_env.step(action)
        running_rewards += r
        running_ep_len += 1
        for j, d in enumerate(done):
            if d:
                ep_cnt += 1
                ep_rets.append(running_rewards[j])
                ep_steps.append(running_ep_len[j])
                running_rewards[j] = running_ep_len[j] = 0
                if ep_cnt >= n_episodes:
                    break
    return dict(
        eval_ret=np.mean(ep_rets),
        eval_len=np.mean(ep_steps),
        eval_ret_std=np.std(ep_rets),
        eval_len_std=np.std(ep_steps),
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
                 eps_min,
                 eps_dec,
                 replace,
                 device,):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.replace_target_cnt = replace
        self.action_space = [i for i in range(self.n_actions)]
        self.learn_step_counter = 0

        self.double = double

        self.memory = ReplayBuffer(mem_size, input_dims, device=device)
        self.device = device

        if len(self.input_dims) == 1:
            self.q_eval = DQN(
                *self.input_dims,
                self.n_actions,
                dueling=dueling,
                linear=linear,
                device=self.device,
            )
        else:
            self.q_eval = AtariDQN(
                self.n_actions,
                linear=linear,
                dueling=dueling,
                device=self.device,
            )

        self.optimizer = T.optim.RMSprop(self.q_eval.parameters(), lr=lr, alpha=0.95, eps=0.01)

        if len(self.input_dims) == 1:
            self.q_next = DQN(
                *self.input_dims,
                self.n_actions,
                dueling=dueling,
                linear=linear,
                device=self.device,
            )
        else:
            self.q_next = AtariDQN(
                self.n_actions,
                linear=linear,
                dueling=dueling,
                device=self.device,
            )

    def choose_action(self, observation, force_random=False, eps=None):
        rnd_action = np.array([
            np.random.choice(self.action_space)
            for _ in range(observation.shape[0])
        ])
        if force_random:
            return rnd_action

        epsilon = self.epsilon if eps is None else eps
        state = T.tensor(observation,
                         dtype=T.float).to(self.q_eval.device)
        q = self.q_eval.forward(state)
        dtm_action = T.argmax(q, -1).cpu().numpy()
        assert dtm_action.shape == (observation.shape[0], ), dtm_action.shape
        mask = np.random.random(size=(observation.shape[0], )) > epsilon
        return mask * dtm_action + (1 - mask) * rnd_action

    def learn(self):
        if self.memory.cur_size < self.batch_size:
            return

        self.optimizer.zero_grad()

        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

        state, actions, new_state, rewards, dones = self.memory.get(self.batch_size)

        states = T.from_numpy(state).to(self.q_eval.device)
        states_ = T.from_numpy(new_state).to(self.q_eval.device)

        indices = np.arange(self.batch_size)

        q_pred = self.q_eval.forward(states)[indices, actions]
        q_next = self.q_next.forward(states_)

        q_next[dones] = 0.0
        if self.double:
            q_eval = self.q_eval.forward(states_)
            max_actions = T.argmax(q_eval, dim=1)
            q_next = q_next[indices, max_actions]
        else:
            q_next = q_next.max(-1).values
        q_target = rewards + self.gamma * q_next

        clipped_error = -1.0 * (q_target - q_pred).clamp(-1, 1)

        # backwards pass
        self.optimizer.zero_grad()
        q_pred.backward(clipped_error.data)
        self.optimizer.step()
        self.learn_step_counter += 1
    
    def decay_epsilon(self):
        self.epsilon = max(self.eps_min, self.epsilon - self.eps_dec)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="CartPole-v1")
    parser.add_argument("--group", type=str)
    parser.add_argument("--trial", type=str)
    parser.add_argument("--dueling", action='store_true')
    parser.add_argument("--double", action="store_true")
    parser.add_argument("--linear", action='store_true')
    parser.add_argument("--wandb", action='store_true')
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    T.backends.cudnn.benchmark = True

    try:
        import wandb
        if args.wandb:
            wandb_run = wandb.init(project='atari_dqn',
                                   config=vars(args),
                                   group=args.group,
                                   name=args.trial)
        else:
            wandb_run = None
    except ModuleNotFoundError:
        wandb_run = None
        logger.info(
            "wandb not installed. Code runs fine without logging to the web.")

    device = T.device(args.device) if T.cuda.is_available() else T.device('cpu')

    train_env = make_env(args.env_name)
    eval_env = SubprocVecEnv(
        [lambda: make_env(args.env_name, eval_=True) for _ in range(32)])

    act_dim = train_env.action_space.n

    total_env_steps = int(200e6)
    eval_interval = int(1e5)
    log_interval = int(1000)
    save_interval = eval_interval * 2

    learning_start = 5e4
    learning_interval = 4

    agent = Agent(gamma=0.99,
                  epsilon=1.0,
                  lr=2.5e-4,
                  input_dims=train_env.observation_space.shape if "NoFrameskip" not in args.env_name else (4, 84, 84),
                  n_actions=train_env.action_space.n,
                  mem_size=int(1e6),
                  dueling=args.dueling,
                  linear=args.linear,
                  double=args.double,
                  eps_min=0.1,
                  batch_size=32,
                  eps_dec=9e-7,
                  replace=10000,
                  device=device,)

    scores = deque(maxlen=100)

    running_rewards = 0
    running_ep_len = 0

    n_env_steps = ep_cnt = 0
    observation = train_env.reset()
    while n_env_steps < total_env_steps:
        action = agent.choose_action(observation[None, :], force_random=(n_env_steps <= learning_start))
        observation_, reward, done, info = train_env.step(action.item())

        running_rewards += reward
        running_ep_len += 1

        agent.memory.put(observation, action, observation_, reward, done)

        if done:
            observation_ = train_env.reset()
            ep_cnt += 1
            scores.append(running_rewards)
            running_rewards = running_ep_len = 0

        observation = observation_

        if n_env_steps >= learning_start and n_env_steps % learning_interval == 0:
            agent.learn()

        n_env_steps += 1
        agent.decay_epsilon()

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
                         eps=agent.epsilon,), n_env_steps)

        if n_env_steps % eval_interval == 0:
            eval_info = eval_dqn(agent, eval_env)
            logger.info(
                "Evaluation Episode Return {:.2f} (± {:.2f})".format(eval_info['eval_ret'], eval_info['eval_ret_std']) + 
                f", Episode Length {eval_info['eval_len']}.")
            if n_env_steps >= learning_start and wandb_run is not None and args.wandb:
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