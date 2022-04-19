
import numpy as np
import torch
import gym

from core import AtariDQN
from env_wrapper import SubprocVecEnv, wrap_deepmind

def make_env(env_name, eval_=False):
    env = gym.make(env_name)
    if "NoFrameskip" in env_name:
        env = wrap_deepmind(env, episode_life=(not eval_), clip_rewards=(not eval_), frame_stack=True, scale=False)
        env = gym.wrappers.TransformObservation(env, lambda x: np.transpose(x, (2, 0, 1)))
    return env

def choose_action(q_net, act_dim, observation, epsilon):
    rnd_action = np.array([
        np.random.choice(act_dim)
        for _ in range(observation.shape[0])
    ])

    state = torch.from_numpy(observation).to(q_net.device) / 255.0
    q = q_net(state)
    dtm_action = torch.argmax(q, -1).cpu().numpy()
    assert dtm_action.shape == (observation.shape[0], ), dtm_action.shape
    mask = np.random.random(size=(observation.shape[0], )) > epsilon
    return mask * dtm_action + (1 - mask) * rnd_action


def eval_dqn(
        q_net,
        act_dim,
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
        action = choose_action(q_net, act_dim, obs, 0.0)
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

device = torch.device(0) if torch.cuda.is_available() else torch.device('cpu')

eval_env = SubprocVecEnv(
    [lambda: make_env("EnduroNoFrameskip-v4", eval_=True) for _ in range(100)])

act_dim = eval_env.action_space.n

q_eval = AtariDQN(
                act_dim,
                linear=False,
                dueling=True,
                device=device,
            )
q_eval.load_state_dict(torch.load(f"endure/double_dqn/results/dqn_double_{int(8.4e6)}.pt", map_location='cpu'))
# double 216.4 (59.2) eps=0.05 293.0 (0.0) eps=0
# dqn 166.6 (20.7) eps=0.05 182.0 (0.0) eps=0

print("start evaluation ...")
eval_info = eval_dqn(q_eval, act_dim, eval_env, n_episodes=100)
print(eval_info)

