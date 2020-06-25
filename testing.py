from metaworld.benchmarks import ML1
from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import core
from logx import EpochLogger
from run_utils import setup_logger_kwargs
from logger import Logger
from metaworld.benchmarks import ML1
import logging
torch.set_num_threads(torch.get_num_threads())


def get_action(o, deterministic=False):
    return ac.act(torch.as_tensor(o, dtype=torch.float32),
                  deterministic)

env = ML1.get_train_tasks('pick-place-v1')  # Create an environment with task `pick_place`
tasks = env.sample_tasks(1)  # Sample a task (in this case, a goal variation)
env.set_task(tasks[0])  # Set task

actor_critic = core.MLPActorCritic
hid = 400
l = 3
ac_kwargs = dict(hidden_sizes=[hid] * l)
gamma = 0.99
seed = 0

torch.manual_seed(seed)
np.random.seed(seed)

#env = ML1.get_train_tasks('reach-v1')  # Create an environment with task `pick_place`
env = ML1.get_train_tasks('pick-place-v1')  # Create an environment with task `pick_place`

tasks = env.sample_tasks(1)  # Sample a task (in this case, a goal variation)
env.set_task(tasks[0])  # Set task

obs_dim = env.observation_space.shape
act_dim = env.action_space.shape[0]

# Action limit for clamping: critically, assumes all dimensions share the same bound!
act_limit = env.action_space.high[0]

# Create actor-critic module and target networks
ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
ac.load_state_dict(torch.load("MetaWorld/MetaWorld_Reach_v1.pth"))
ac.eval()

steps = 0
ep_len = 0
ep_ret = 0
o, ep_ret, ep_len = env.reset(), 0, 0
while True:
    a = get_action(o)
    steps+=1
    ep_len+=1
    o2, r, d, _ = env.step(a)
    ep_ret += r
    o = o2
    env.render()
    if d or (ep_len == 150):
        print("return: ", ep_ret)
        o, ep_ret, ep_len = env.reset(), 0, 0

# obs, reward, done, info = env.step(a)  # Step the environoment with the sampled random action
