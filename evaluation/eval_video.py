import numpy as np
import moviepy.video.io.ImageSequenceClip

from evaluation.eval_utils import to_img_padded, to_moviepy_list, init_evaluation


def create_video(env, agent, pad, filename, fps, seconds):
    vid_imgs = []
    observation, ep_reward, done = env.reset(), 0, False
    count = 0
    while count < fps * seconds:
        action, generated_next_observation = agent.act(observation, get_generated=True)
        next_observation, reward, done, info = env.step(action)
        imgs = [to_img_padded(observation, pad),
                to_img_padded(generated_next_observation, pad),
                to_img_padded(next_observation, pad)]
        vid_imgs.append(np.concatenate(to_moviepy_list(imgs), axis=1))
        count += 1
        observation = next_observation
        if done:
            observation, ep_reward, done = env.reset(), 0, False
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(vid_imgs, fps=60)
    clip.write_videofile(f"{filename}.mp4")


def main():
    pad = [(5, 5), (5, 5), (0, 0)]
    train_env, test_env, agent, agent_folder = init_evaluation(model="noisy_linear", use_backgrounds=False)
    create_video(train_env, agent, pad, f"{agent_folder}/train_envs", fps=60, seconds=30)
    create_video(test_env, agent, pad, f"{agent_folder}/test_envs", fps=60, seconds=30)


main()
