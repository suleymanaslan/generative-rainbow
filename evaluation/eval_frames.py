import numpy as np
import imageio

from evaluation.eval_utils import to_img_padded, format_img, init_evaluation


def save_frames(env, agent, pad, folder):
    observation, ep_reward, done = env.reset(), 0, False
    count = 0
    while not done:
        action, generated_next_observation = agent.act(observation, get_generated=True)
        next_observation, reward, done, info = env.step(action)
        imgs = [to_img_padded(observation, pad),
                to_img_padded(generated_next_observation, pad),
                to_img_padded(next_observation, pad)]
        plot_img = np.concatenate(format_img(imgs), axis=1)
        count += 1
        imageio.imwrite(f"{folder}/{count:04d}.png", plot_img)
        observation = next_observation


def main():
    pad = [(5, 5), (5, 5), (0, 0)]
    train_env, test_env, agent, agent_folder = init_evaluation(use_backgrounds=False)
    save_frames(train_env, agent, pad, f"{agent_folder}/frames/train")
    save_frames(test_env, agent, pad, f"{agent_folder}/frames/test")


main()
