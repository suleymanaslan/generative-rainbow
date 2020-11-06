import numpy as np
import moviepy.video.io.ImageSequenceClip

from evaluation.eval_utils import to_img, to_moviepy_list, init_evaluation


def create_video(env, agent, filename, fps, seconds):
    vid_imgs = []
    observation, ep_reward, done = env.reset(), 0, False
    count = 0
    while count < fps * seconds:
        action, generated_next_observation = agent.act(observation, get_generated=True)
        next_observation, reward, done, info = env.step(action)
        imgs = [to_img(observation), to_img(generated_next_observation), to_img(next_observation)]
        vid_imgs.append(np.concatenate(to_moviepy_list(imgs), axis=1))
        count += 1
        observation = next_observation
        if done:
            observation, ep_reward, done = env.reset(), 0, False
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(vid_imgs, fps=60)
    clip.write_videofile(f"{filename}.mp4")


def main():
    train_env, test_env, agent, agent_folder = init_evaluation(use_backgrounds=False)
    create_video(train_env, agent, f"{agent_folder}/train_envs", fps=60, seconds=30)
    create_video(test_env, agent, f"{agent_folder}/test_envs", fps=60, seconds=30)


main()
