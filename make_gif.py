from matplotlib import animation
import matplotlib.pyplot as plt
import gym
import torch
import numpy as np
import os

from core import AtariDQN
from env_wrapper import wrap_deepmind


def make_env(env_name, eval_=False):
    env = gym.make(env_name)
    if "NoFrameskip" in env_name:
        env = wrap_deepmind(env,
                            episode_life=(not eval_),
                            clip_rewards=(not eval_),
                            frame_stack=True,
                            scale=False)
        env = gym.wrappers.TransformObservation(
            env, lambda x: np.transpose(x, (2, 0, 1)))
    return env


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
                                   interval=50)
    anim.save(os.path.join(path, file_name), writer='imagemagick', fps=60)


env = make_env("EnduroNoFrameskip-v4", eval_=True)

obs = env.reset()
frames = []

device = torch.device(0) if torch.cuda.is_available() else torch.device('cpu')

act_dim = env.action_space.n

q = AtariDQN(
    act_dim,
    linear=True,
    dueling=False,
    device=device,
)
q.load_state_dict(
    torch.load(f"endure/linear/results/dqn_linear_{int(2.8e6)}.pt",
               map_location='cpu'))


def choose_action(q_net, act_dim, observation, epsilon):
    rnd_action = np.array(
        [np.random.choice(act_dim) for _ in range(observation.shape[0])])

    state = torch.from_numpy(observation).to(q_net.device) / 255.0
    q = q_net(state)
    dtm_action = torch.argmax(q, -1).cpu().numpy()
    assert dtm_action.shape == (observation.shape[0], ), dtm_action.shape
    mask = np.random.random(size=(observation.shape[0], )) > epsilon
    return mask * dtm_action + (1 - mask) * rnd_action


for t in range(1000):
    # print(env.render(mode='rgb_array'))
    frames.append(env.render(mode='rgb_array'))
    action = choose_action(q, act_dim, obs[None, :], 0.05)
    obs, _, done, _ = env.step(action.item())
    if done:
        break

env.close()
save_frames_as_gif(frames, file_name='linear_0.3.gif')