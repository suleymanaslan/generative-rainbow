from env import StarPilotEnv
from agent import Agent
from replay_memory import ReplayMemory
from utils import Trainer
import torch
import numpy as np

torch.manual_seed(828)
np.random.seed(828)

env = StarPilotEnv(action_size=15, history_length=4,
                   num_levels=10, start_level=0, distribution_mode="easy",
                   use_backgrounds=False)
train_env = StarPilotEnv(action_size=15, history_length=4,
                         num_levels=env.num_levels, start_level=env.start_level,
                         distribution_mode=env.distribution_mode,
                         use_backgrounds=env.use_backgrounds)
test_env = StarPilotEnv(action_size=15, history_length=4,
                        num_levels=20, start_level=int(100e3),
                        distribution_mode=env.distribution_mode,
                        use_backgrounds=env.use_backgrounds)
agent = Agent(env, atoms=51, v_min=-20.0, v_max=20.0, batch_size=64, multi_step=3,
              discount=0.99, norm_clip=10.0, lr=5e-4, adam_eps=1.5e-4, hidden_size=256,
              noisy_std=0.1)
mem = ReplayMemory(int(50e3), env.window, agent.discount, agent.n,
                   priority_weight=0.4, priority_exponent=0.5)
trainer = Trainer(max_steps=int(1800e3), replay_frequency=6, reward_clip=5.0,
                  learning_start_step=int(5e3), target_update=int(2e3),
                  gan_steps=int(150e3), eval_steps=int(50e3), plot_steps=int(25e3),
                  training_mode="joint")

trainer.train(env, train_env, test_env, agent, mem, file="gan_rl.py")
trainer.save(agent)
