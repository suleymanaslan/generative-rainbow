from env import StarPilotEnv
from agent import Agent
from replay_memory import ReplayMemory
from utils import Trainer
import torch
import numpy as np

torch.manual_seed(828)
np.random.seed(828)

env = StarPilotEnv(action_size=15, history_length=16,
                   num_levels=10, start_level=0, distribution_mode="easy",
                   use_backgrounds=False, view_mode="rgb")
train_env = StarPilotEnv(action_size=15, history_length=env.window,
                         num_levels=env.num_levels, start_level=env.start_level,
                         distribution_mode=env.distribution_mode,
                         use_backgrounds=env.use_backgrounds, view_mode=env.view_mode)
test_env = StarPilotEnv(action_size=15, history_length=env.window,
                        num_levels=20, start_level=int(100e3),
                        distribution_mode=env.distribution_mode,
                        use_backgrounds=env.use_backgrounds, view_mode=env.view_mode)
agent = Agent(env, atoms=51, v_min=-20.0, v_max=20.0, batch_size=64, multi_step=3,
              discount=0.99, norm_clip=10.0, lr=5e-4, adam_eps=1.5e-4, hidden_size=256,
              noisy_std=0.1, gan_alpha=1e-4, training_mode="branch")
mem = ReplayMemory(int(50e3), env, agent.discount, agent.n,
                   priority_weight=0.4, priority_exponent=0.5)
trainer = Trainer(max_steps=int(3e6), replay_frequency=16, reward_clip=5.0,
                  learning_start_step=int(5e3), target_update=int(2e3),
                  gan_steps=int(400e3), gan_scale_steps=int(400e3),
                  eval_steps=int(50e3), plot_steps=int(25e3), training_mode=agent.training_mode)

trainer.train(env, train_env, test_env, agent, mem, file="gan_rl.py")
trainer.save(agent)
