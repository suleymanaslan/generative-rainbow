from env import StarPilotEnv
from ppo.buffer import Buffer
from ppo.actor_critic import ActorCritic
from ppo.agent import Agent
from ppo.utils import PPOTrainer
import torch
import numpy as np

torch.manual_seed(828)
np.random.seed(828)

env = StarPilotEnv(history_length=16, num_levels=10,
                   start_level=0, distribution_mode="easy",
                   use_backgrounds=False, view_mode="rgb")
train_env = StarPilotEnv(history_length=env.window, num_levels=env.num_levels,
                         start_level=env.start_level, distribution_mode=env.distribution_mode,
                         use_backgrounds=env.use_backgrounds, view_mode=env.view_mode)
test_env = StarPilotEnv(history_length=env.window, num_levels=20,
                        start_level=int(100e3), distribution_mode=env.distribution_mode,
                        use_backgrounds=env.use_backgrounds, view_mode=env.view_mode)
buffer = Buffer(size=1024, obs_shape=env.obs_shape, gamma=0.99, lam=0.97)
actor_critic = ActorCritic(env, action_size=env.action_size)
agent = Agent(buffer, actor_critic, policy_lr=3e-4, value_lr=1e-3,
              policy_train_iter=64, value_train_iter=64)
trainer = PPOTrainer(max_steps=int(3e6), plot_steps=int(25e3), eval_steps=int(50e3))

trainer.train(env, train_env, test_env, agent, buffer, actor_critic, file="gan_ppo.py")
trainer.save(agent)
