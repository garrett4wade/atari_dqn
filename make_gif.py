from matplotlib import animation
import matplotlib.pyplot as plt
import gym
import torch
import numpy as np
import os

from core import AtariDQN
from main import make_env


def save_frames_as_gif(frames, path=".", file_name='test.gif'):
    plt.figure(figsize=(frames[0].shape[1] / 72, frames[0].shape[0] / 72),
               dpi=72)
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(),
                                   animate,
                                   frames=len(frames),
                                   interval=5)
    anim.save(os.path.join(path, file_name), writer='imagemagick')


@torch.no_grad()
def choose_action(q_net, act_dim, observation, epsilon):
    rnd_action = np.array(
        [np.random.choice(act_dim) for _ in range(observation.shape[0])])

    state = torch.from_numpy(observation).to(q_net.device) / 255.0
    q = q_net(state)
    dtm_action = torch.argmax(q, -1).cpu().numpy()
    assert dtm_action.shape == (observation.shape[0], ), dtm_action.shape
    mask = np.random.random(size=(observation.shape[0], )) > epsilon
    return mask * dtm_action + (1 - mask) * rnd_action


env = make_env("EnduroNoFrameskip-v4", eval_=True)

device = torch.device(0) if torch.cuda.is_available() else torch.device('cpu')

act_dim = env.action_space.n

ckpt_steps = [int(0.4e6), int(1.6e6), int(3.6e6), int(5.6e6)]
ckpt_prefixes = [
    "/root/dqn/results/dqn_", "/root/double_dqn/results/dqn_double_",
    "/root/dueling/results/dqn_double_", "/root/linear/results/dqn_linear_",
    "/root/linear_double/results/dqn_linear_double_"
]
tags = ['dqn', 'double_dqn', 'dd_dqn', 'linear', 'linear_double']

for gif_idx, step in enumerate(ckpt_steps):
    for prefix, tag in zip(ckpt_prefixes, tags):
        q = AtariDQN(
            act_dim,
            linear=("linear" in prefix),
            dueling=("dueling" in prefix),
            device=device,
        )
        q.load_state_dict(torch.load(prefix + f"{step}.pt",
                                     map_location='cpu'))

        obs = env.reset()
        frames = []

        done = False
        while not done:
            frames.append(env.render(mode='rgb_array'))
            action = choose_action(q, act_dim, obs[None, :], 0.05)
            obs, _, done, _ = env.step(action.item())

        env.close()
        save_frames_as_gif(frames, file_name=f'{tag}_{gif_idx}.gif')
        print(f"saved gif file {f'{tag}_{gif_idx}.gif'}!")
