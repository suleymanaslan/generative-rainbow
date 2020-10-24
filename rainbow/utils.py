from datetime import datetime

from utils import Trainer


class RainbowTrainer(Trainer):
    def __init__(self, max_steps, replay_frequency, reward_clip, learning_start_step,
                 target_update, gan_steps, gan_scale_steps, eval_steps, plot_steps, training_mode):
        super(RainbowTrainer, self).__init__(max_steps, plot_steps, eval_steps)
        self.replay_frequency = replay_frequency
        self.reward_clip = reward_clip
        self.learning_start_step = learning_start_step
        self.target_update = target_update
        self.training_mode = training_mode
        self.gan_steps = gan_steps
        self.gan_scale_steps = gan_scale_steps

    def train(self, env, train_env, test_env, agent, mem, mem_generated, file=None):
        self._init_training(file)
        priority_weight_increase = (1 - mem.priority_weight) / (self.max_steps - self.learning_start_step)
        finished = False
        episode = 0
        steps = 0
        generated_next_observation = None
        while not finished:
            observation, ep_reward, done = env.reset(), 0, False
            generated_observation = observation
            while not done:
                if agent.scale == agent.max_scale:
                    action, generated_next_observation = agent.act(observation, get_generated=True)
                    next_observation, reward, done, info = env.step(action, generated_next_observation)
                    generated_next_observation = info["generated_observation"]
                else:
                    action = agent.act(observation)
                    next_observation, reward, done, info = env.step(action)
                    generated_observation = next_observation
                ep_reward += reward
                steps += 1
                if steps % self.eval_steps == 0:
                    self.eval(train_env, test_env, agent, steps)
                if steps % self.plot_steps == 0:
                    self.save(agent)
                if steps >= self.max_steps:
                    finished = True
                    break
                if steps % self.replay_frequency == 0:
                    agent.reset_noise()
                if self.reward_clip > 0:
                    reward = max(min(reward, self.reward_clip), -self.reward_clip) / self.reward_clip
                mem.append(observation, action, reward, done)
                if agent.scale == agent.max_scale:
                    mem_generated.append(generated_observation, action, reward, done)
                if steps >= self.learning_start_step:
                    mem.priority_weight = min(mem.priority_weight + priority_weight_increase, 1)
                    mem_generated.priority_weight = mem.priority_weight
                    if self.training_mode == "joint":
                        if steps % self.replay_frequency == 0:
                            agent.learn_joint(mem, self)
                    elif self.training_mode == "separate":
                        if steps % self.gan_steps == 0:
                            agent.learn_gan(mem, self,
                                            repeat=(self.gan_steps * agent.steps_per_scale) // self.gan_scale_steps)
                        if steps % self.replay_frequency == 0:
                            agent.learn(mem, self)
                    elif self.training_mode == "gan_feat":
                        if steps % self.replay_frequency == 0:
                            agent.learn_gan_feat(mem, self)
                    elif self.training_mode == "dqn_only":
                        if steps % self.replay_frequency == 0:
                            agent.learn(mem, self)
                    elif self.training_mode == "branch":
                        if steps % self.replay_frequency == 0:
                            if steps >= self.learning_start_step * 2 + \
                                    agent.steps_per_scale * self.replay_frequency * agent.max_scale:
                                agent.learn_branch_generated(mem, mem_generated, self)
                            else:
                                agent.learn_branch(mem, self)
                    elif self.training_mode == "gan_only":
                        agent.learn_gan(mem, self, repeat=1)
                    else:
                        raise NotImplementedError
                    if steps % self.target_update == 0:
                        agent.update_target_net()
                observation = next_observation
                if generated_next_observation is not None:
                    generated_observation = generated_next_observation
            episode += 1
            self.ep_rewards.append(ep_reward)
            self.ep_steps.append(steps)
            if episode == 1 or episode % 100 == 0:
                self.print_and_log(f"{datetime.now()}, episode:{episode:5d}, step:{steps:6d}, reward:{ep_reward:4.1f}")
        self.print_and_log(f"{datetime.now()}, end training")

    @staticmethod
    def _eval(env, num_levels, agent):
        eval_reward = 0
        for _ in range(num_levels):
            observation, ep_reward, done = env.reset(), 0, False
            while not done:
                action = agent.act(observation)
                next_observation, reward, done, info = env.step(action)
                ep_reward += reward
                observation = next_observation
            eval_reward += ep_reward
        eval_reward /= num_levels
        return eval_reward
