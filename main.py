import argparse
import gym
import logging
import numpy as np
import torch

from core import ReplayBuffer, AtariDQN, MLPDQN

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
        )
        env = gym.wrappers.FrameStack(env, 4)
        if eval_:
            # env = gym.wrappers.RecordVideo(env, './results')
            pass
    return env


def dqn_loss(q1_net,
             q2_net,
             s,
             a,
             n_s,
             r,
             d,
             gamma,
             double,
             device=torch.device("cuda:0")):
    s = torch.from_numpy(pixel_to_float(s)).to(device)
    n_s = torch.from_numpy(pixel_to_float(n_s)).to(device)
    a = torch.from_numpy(a).to(device)
    r = torch.from_numpy(r).to(device)
    d = torch.from_numpy(d).to(device)

    if double and np.random.rand() < 0.5:
        q1_net, q2_net = q2_net, q1_net

    with torch.no_grad():
        if double:
            n_a = q1_net(n_s).argmax(-1, keepdim=True)
            n_qa = q2_net(n_s).gather(-1, n_a)
        else:
            n_qa = q2_net(n_s).max(-1, keepdim=True).values
        assert n_qa.shape[-1] == 1
        # n_qa = q2_net.fc.denormalize(n_qa)

        q_target = r + gamma * (1 - d) * n_qa.squeeze(-1)
        # q_target = q_target.unsqueeze(-1)

        # q1_net.fc.update(q_target)

        # mu, sigma = q1_net.fc.mean_std()
        # logger.debug(f"Update PopArt to Mean {mu.item()} Std {sigma.item()}")

        # q_target = q1_net.fc.normalize(q_target)
        # q_target = q_target.squeeze(-1)

    qa = q1_net(s).gather(-1, a.unsqueeze(-1).long()).squeeze(-1)
    assert len(qa.shape) == 1 and len(q_target.shape) == 1, (qa.shape,
                                                             q_target.shape)
    loss = (q_target - qa)**2
    return loss.mean()


def eval_dqn(q_net,
             eval_env,
             n_episodes=20,
             device=torch.device("cuda:0"),
             deterministic=False,
             eps=0.05):
    ep_cnt = total_ep_step = total_ep_ret = total_time = 0
    while ep_cnt < n_episodes:
        obs = eval_env.reset()
        done = False
        while not done:
            obs_net = torch.from_numpy(
                pixel_to_float(obs)).unsqueeze(0).to(device)
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


def train_dqn(q1_net,
              q2_net,
              buffer,
              train_env,
              eval_env,
              use_wandb,
              eval_interval=500,
              log_interval=500,
              warmup_steps=1000,
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

    obs = train_env.reset()
    for _ in range(warmup_steps):
        act = train_env.action_space.sample()
        obs_, rew, don, info = train_env.step(act)
        buffer.put(obs, act, obs_, rew, don)
        if don:
            obs_ = train_env.reset()
        obs = obs_

    env_frames = warmup_steps
    step = 0

    eps = eps_start

    while env_frames < total_env_steps:
        log_info = {}

        obs_net = torch.from_numpy(pixel_to_float(obs)).unsqueeze(0).to(device)
        if np.random.rand() < eps:
            act = train_env.action_space.sample()
        else:
            act = q1_net(obs_net).argmax(-1).item()
        obs_, rew, don, info = train_env.step(act)

        buffer.put(obs, act, obs_, rew, don)

        if don:
            obs_ = train_env.reset()
            logger.info(
                f"Train env episode done with episode return"
                f" {info['episode']['r']} and episode length {info['episode']['l']}"
            )
            log_info = dict(episode_return=info['episode']['r'],
                            episode_length=info['episode']['l'])

        obs = obs_

        s, a, n_s, r, d = buffer.get(bs)
        q1_optimizer.zero_grad()
        if q2_optimizer is not None:
            assert double
            q2_optimizer.zero_grad()

        loss = dqn_loss(q1_net, q2_net, s, a, n_s, r, d, gamma, double)
        loss.backward()

        q1_optimizer.step()
        if q2_optimizer is not None:
            q2_optimizer.step()

        log_info = dict(loss=loss.item(), **log_info)

        eps = max(eps_end, eps - eps_decay_per_step)

        step += 1
        env_frames += 1

        if not double and step % target_update_interval == 0:
            q2_net.load_state_dict(q1_net.state_dict())
            logger.info("Update target network.")

        if step % eval_interval == 0:
            eval_info = eval_dqn(q1_net, eval_env)
            log_info = dict(eval_info, **log_info)

        if step % log_interval == 0:
            if use_wandb:
                wandb.log(log_info, step=env_frames)
            torch.save(q1_net.state_dict(), f"./results/ckpt_{step}.pt")
            logger.info(
                f"Environment steps {int(env_frames)}/{total_env_steps}, log info {log_info}"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name",
                        type=str,
                        default="SpaceInvadersNoFrameskip-v4")
    parser.add_argument("--dueling", action='store_true')
    parser.add_argument("--double", action="store_true")
    parser.add_argument("--linear", action='store_true')
    parser.add_argument("--wandb", action='store_true')
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

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

    device = torch.device("cuda:0")

    train_env = make_env(args.env_name)
    eval_env = make_env(args.env_name, eval_=True)

    act_dim = train_env.action_space.n
    buffer = ReplayBuffer(int(1e6), train_env.observation_space.shape, act_dim)

    if "NoFrameskip" in args.env_name:
        q1_net = AtariDQN(
            [3, 3, 3, 3],
            act_dim,
            act_fn=(lambda: torch.nn.ReLU(inplace=True)
                    if not args.linear else torch.nn.Identity()),
            dueling=args.dueling,
        ).to(device)
        q2_net = AtariDQN(
            [3, 3, 3, 3],
            act_dim,
            act_fn=(lambda: torch.nn.ReLU(inplace=True)
                    if not args.linear else torch.nn.Identity()),
            dueling=args.dueling,
        ).to(device)
    else:
        q1_net = MLPDQN(
            train_env.observation_space.shape[0],
            act_dim,
            act_fn=(lambda: torch.nn.ReLU(inplace=True)
                    if not args.linear else torch.nn.Identity()),
        ).to(device)
        q2_net = MLPDQN(
            train_env.observation_space.shape[0],
            act_dim,
            act_fn=(lambda: torch.nn.ReLU(inplace=True)
                    if not args.linear else torch.nn.Identity()),
        ).to(device)
    if not args.double:
        q2_net.load_state_dict(q1_net.state_dict())

    train_dqn(q1_net,
              q2_net,
              buffer,
              train_env,
              eval_env,
              use_wandb=args.wandb)

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()