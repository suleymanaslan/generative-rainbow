from datetime import datetime

from utils import Trainer


class PPOTrainer(Trainer):
    def __init__(self, max_steps, plot_steps, eval_steps):
        super(PPOTrainer, self).__init__(max_steps, plot_steps, eval_steps)

    def train(self, env, train_env, test_env, agent, buffer, actor_critic, file=None):
        self._init_training(file)
        finished = False
        episode = 0
        steps = 0
        observation, ep_reward = env.reset(), 0
        while not finished:
            for buffer_step in range(buffer.size):
                action, value, log_prob_action = actor_critic.step(observation)
                next_observation, reward, done, info = env.step(action.item())
                ep_reward += reward
                steps += 1
                if steps % self.eval_steps == 0:
                    self.eval(train_env, test_env, actor_critic, steps)
                if steps % self.plot_steps == 0:
                    self.save(agent)
                buffer.store(observation, action, reward, value, log_prob_action)
                observation = next_observation

                if done:
                    value = 0
                    buffer.finish_path(value)
                    episode += 1
                    self.ep_rewards.append(ep_reward)
                    self.ep_steps.append(steps)
                    if episode == 1 or episode % 50 == 0:
                        self.print_and_log(
                            f"{datetime.now()}, episode:{episode:4d}, step:{steps:5d}, reward:{ep_reward:10.4f}")
                    observation, ep_reward = env.reset(), 0
                elif buffer_step == buffer.size - 1:
                    _, value, _ = actor_critic.step(observation)
                    buffer.finish_path(value.cpu().numpy())

            agent.learn()

            if steps >= self.max_steps:
                finished = True
        self.print_and_log(f"{datetime.now()}, end training")

    @staticmethod
    def _eval(env, num_levels, actor_critic):
        eval_reward = 0
        for _ in range(num_levels):
            observation, ep_reward, done = env.reset(), 0, False
            while not done:
                action, value, log_prob_action = actor_critic.step(observation)
                next_observation, reward, done, info = env.step(action.item())
                ep_reward += reward
                observation = next_observation
            eval_reward += ep_reward
        eval_reward /= num_levels
        return eval_reward
