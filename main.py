import argparse
import gym
import logging
import numpy as np
import torch

logger = logging.getLogger()
logger.setLevel(logging.INFO)

try:
    import wandb
except ModuleNotFoundError:
    logger.info(
        "wandb not installed. Code runs fine without logging to the web.")

from core import ReplayBuffer, AtariDQN


def pixel_to_float(x):
    return x.astype(np.float32) / 255


def channel_last(x):
    return np.transpose(x, [1, 2, 0])


def make_env(env_name, eval_=False):
    env = gym.make(env_name)
    env = gym.wrappers.AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=4,
        screen_size=84,
        scale_obs=False,
    )
    env = gym.wrappers.OrderEnforcing(env)
    env = gym.wrappers.FrameStack(env, 4)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.TransformObservation(env, channel_last)
    if eval_:
        env = gym.wrappers.TransformObservation(env, pixel_to_float)
        env = gym.wrappers.RecordVideo(env, './results')
    return env


def dqn_loss(q1_net, q2_net, s, a, n_s, r, d, gamma, double):
    if double and np.random.rand() < 0.5:
        q1_net, q2_net = q2_net, q1_net

    with torch.no_grad():
        if double:
            n_a = q1_net(s).argmax(-1)
            n_qa = q2_net(n_s).gather(-1, n_a.unsqueeze(-1))
        else:
            n_qa = q2_net(n_s).max(-1).values

    qa = q1_net(s).gather(-1, a.unsqueeze(-1))
    loss = (r + gamma * (1 - d) * n_qa - qa)**2
    return loss.mean()


def eval_dqn(q_net, eval_env, n_episodes=20):
    ep_cnt = total_ep_step = total_ep_ret = 0
    while ep_cnt < n_episodes:
        obs = eval_env.reset()
        done = False
        ep_step = ep_ret = 0
        while not done:
            obs_net = torch.from_numpy(pixel_to_float(obs)).to(device)
            act = q_net(obs_net).argmax(-1).item()
            obs, r, done, _ = env.step(act)
            ep_step += 1
            ep_ret += r
        ep_cnt += 1
        total_ep_ret += ep_ret
        total_ep_step += ep_step
    return dict(eval_episode_return=total_ep_ret / n_episodes,
                eval_episode_length=total_ep_step / n_episodes)


def train_dqn(q1_net,
              q2_net,
              buffer,
              train_env,
              eval_env,
              eval_interval=100,
              log_interval=100,
              total_env_steps=int(5e6),
              lr=1e-4,
              bs=32,
              target_update_interval=int(1e4),
              eps_start=1.0,
              eps_end=0.05,
              eps_decay_per_step=1e-3,
              gamma=0.99,
              double=True,
              device=torch.device("cuda:0")):
    q1_optimizer = torch.optim.Adam(q1_net.parameters(), lr=lr)
    if double:
        q2_optimizer = torch.optim.Adam(q2_net.parameters(), lr=lr)
    else:
        q2_optimizer = None

    env_frames = 0
    step = 0
    obs = train_env.reset()
    eps = eps_start
    while env_frames < total_env_steps:
        obs_net = torch.from_numpy(pixel_to_float(obs)).to(device)
        if np.random.rand() < eps:
            act = train_env.action_space.sample()
        else:
            act = q1_net(obs_net).argmax(-1).item()
        obs_, rew, don, info = train_env.step(act)

        buffer.put(obs, a, obs_, rew, don)

        if d:
            obs_ = train_env.reset()

        obs = obs_

        sample = buffer.get(bs)
        if sample is not None:
            s, a, n_s, r, d = sample
            q1_optimizer.zero_grad()
            if q2_optimizer is not None:
                assert double
                q2_optimizer.zero_grad()

            loss = dqn_loss(q1_net, q2_net, s, a, n_s, r, d, gamma, double)
            loss.backward()

            q1_optimizer.step()
            if q2_optimizer is not None:
                q2_optimizer.step()

        env_frames += 1
        step += 1

        eps = max(eps_end, eps - eps_decay_per_step)

        if not double and step % target_update_interval == 0:
            q2_net.load_state_dict(q1_net.state_dict())

        log_info = dict(loss=loss)

        if step % eval_interval == 0:
            eval_info = eval_dqn(q1_net, eval_nevs)
            log_info = dict(eval_info, **log_info)

        if step % log_interval == 0:
            wandb.log(log_info, step=env_frames)
            torch.save(q1_net.state_dict(), f"./results/ckpt_{step}.pt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name",
                        type=str,
                        default="SpaceInvadersNoFrameskip-v4")
    parser.add_argument("--buffer_maxsize", type=int, default=int(1e6))
    parser.add_argument("--dueling", action='store_true')
    parser.add_argument("--double", action="store_true")
    parser.add_argument("--linear", action='store_true')
    args = parser.parse_args()

    device = torch.device("cuda:0")

    train_env = make_env(args.env_name)
    eval_env = make_env(args.env_name, eval_=True)
    obs = train_env.reset()
    print(obs.shape, obs.dtype)

    act_dim = train_env.action_space.n
    buffer = ReplayBuffer(int(1e6), (84, 84, 4), act_dim)

    q1_net = AtariDQN([3, 3, 3, 3],
                      act_dim,
                      act_fn=(lambda: torch.nn.ReLU(inplace=True)
                              if not args.linear else torch.nn.Identity),
                      dueling=args.dueling).to(device)
    q2_net = AtariDQN([3, 3, 3, 3],
                      act_dim,
                      act_fn=(lambda: torch.nn.ReLU(inplace=True)
                              if not args.linear else torch.nn.Identity),
                      dueling=args.dueling).to(device)
    if not double:
        q2_net.load_state_dict(q1_net)
    
    eval_dqn(q1_net, eval_env)

    # train_dqn(q1_net, q2_net, buffer, train_env, eval_env)


if __name__ == "__main__":
    main()