from metaworld.benchmarks import ML1
import time
import time
from copy import deepcopy
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

from metaworld.benchmarks import ML1

def get_action(o, deterministic=False):
    return ac.act(torch.as_tensor(o, dtype=torch.float32),
                  deterministic)

env = ML1.get_train_tasks('pick-place-v1')  # Create an environment with task `pick_place`
tasks = env.sample_tasks(1)  # Sample a task (in this case, a goal variation)
env.set_task(tasks[0])  # Set task

steps_per_epoch=4000
epochs = 50

total_steps = steps_per_epoch * epochs
start_time = time.time()
o, ep_ret, ep_len = env.reset(), 0, 0
start_steps = 10000
hid = 256
l = 2
obs_dim = env.observation_space.shape
act_dim = env.action_space.shape[0]
ac_kwargs = dict(hidden_sizes=[hid] * l)

# Action limit for clamping: critically, assumes all dimensions share the same bound!
act_limit = env.action_space.high[0]
actor_critic = core.MLPActorCritic

# Create actor-critic module and target networks
ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
ac_targ = deepcopy(ac)

# Main loop: collect experience in env and update/log each epoch
for t in range(total_steps):
    if t > start_steps:
        a = get_action(o)
    else:
        a = env.action_space.sample()

    # Step the env
    o2, r, d, _ = env.step(a)
    print("got reward")
    print(d)
    print(t)








obs = env.reset()  # Reset environment
a = env.action_space.sample()  # Sample an action
obs, reward, done, info = env.step(a)  # Step the environoment with the sampled random action

