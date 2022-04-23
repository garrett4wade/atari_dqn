import numpy as np
import torch
import gym

from core import AtariDQN
from env_wrapper import SubprocVecEnv
from main import make_env


def choose_action(q_net, act_dim, observation, epsilon):
    rnd_action = np.array(
        [np.random.choice(act_dim) for _ in range(observation.shape[0])])

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
        action = choose_action(q_net, act_dim, obs, 0.05)
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
    linear=True,
    dueling=False,
    device=device,
)
q_eval.load_state_dict(
    torch.load(f"/root/linear_double/results/dqn_linear_double_{int(5.6e6)}.pt",
               map_location='cpu'))
# double dueling 1095.3 (173.4)
# dqn 967.5 (155.2)
# doule dqn 596.9 (147.7)
# linear 88.5 (24.0)
# linear double 81.3 (28.4)
print("start evaluation ...")
eval_info = eval_dqn(q_eval, act_dim, eval_env, n_episodes=100)
print(eval_info)

eval_env.close()
