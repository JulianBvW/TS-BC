import cv2
import gym
import torch
import minerl
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival

from EpisodeActions import EpisodeActions
from LatentSpaceMineCLIP import LatentSpaceMineCLIP, load_mineclip, AGENT_RESOLUTION, SLIDING_WINDOW_SIZE

TEXT_GOAL = 'gather wood'

MAX_FRAMES = 1*60*20
ENV_KWARGS = dict(
    fov_range=[70, 70],
    frameskip=1,
    gamma_range=[2, 2],
    guiscale_range=[1, 1],
    resolution=[640, 360],
    cursor_size_range=[16.0, 16.0],
)

frame_counter = 0
nearest_idx = 0

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter('agent_recording.mp4', fourcc, 20, (640, 360))

mineclip_model = load_mineclip()
sliding_window_frames = []
follow_frame = 0
log = []

episode_actions = EpisodeActions()
episode_actions.load()

latent_space_mineclip = LatentSpaceMineCLIP()
latent_space_mineclip.load()

text_latent = mineclip_model.encode_text(TEXT_GOAL)[0].detach()
print('### TEXT LATENT of shape', text_latent.shape)

goal_distances = latent_space_mineclip.get_distances(text_latent)
print('### Goal distances of shape', goal_distances.shape)
print(goal_distances[len(goal_distances)-10:])
import pandas as pd
print(pd.Series(goal_distances.to('cpu')).describe())

nearest_idx = latent_space_mineclip.get_nearest(text_latent)
nearest_idx = nearest_idx.to('cpu').item()

episode = 0
while episode+1 < len(episode_actions.episode_starts) and int(episode_actions.episode_starts[episode+1][1]) <= nearest_idx:
    episode += 1
episode_id, episode_start = episode_actions.episode_starts[episode]
episode_start = int(episode_start)

episode_frame = nearest_idx - episode_start + SLIDING_WINDOW_SIZE - 1
print(f'[Frame {frame_counter} ({frame_counter // 20 // 60}:{(frame_counter // 20) % 60})] Found nearest in {episode_id} at frame {episode_frame} ({episode_frame // 20 // 60}:{(episode_frame // 20) % 60})')

ROLLING_SIZE = 20*20

goal_distances_padding = F.pad(goal_distances, (0, ROLLING_SIZE-1), 'constant', float('inf'))
goal_distances_rolling = goal_distances_padding.unfold(0, ROLLING_SIZE, 1)
future_goal_distances = goal_distances_rolling.min(1).values

print(future_goal_distances[len(future_goal_distances)-10:])
print(pd.Series(future_goal_distances.to('cpu')).describe())


def search(sliding_window_frames):
    global log, frame_counter

    frame_window = torch.from_numpy(np.array(sliding_window_frames)).unsqueeze(0).to('cuda')
    latent = mineclip_model.encode_video(frame_window)[0]
    del(frame_window)

    # nearest_idx = latent_space_mineclip.get_nearest(latent)
    # nearest_idx = nearest_idx.to('cpu').item()

    bla = future_goal_distances + latent_space_mineclip.get_distances(latent)
    nearest_idx = bla.argmin().to('cpu').item()

    episode = 0
    while episode+1 < len(episode_actions.episode_starts) and int(episode_actions.episode_starts[episode+1][1]) <= nearest_idx:
        episode += 1
    episode_id, episode_start = episode_actions.episode_starts[episode]
    episode_start = int(episode_start)

    episode_frame = nearest_idx - episode_start + SLIDING_WINDOW_SIZE - 1
    log.append(f'[Frame {frame_counter} ({frame_counter // 20 // 60}:{(frame_counter // 20) % 60})] Found nearest in {episode_id} at frame {episode_frame} ({episode_frame // 20 // 60}:{(episode_frame // 20) % 60})')

    return nearest_idx

def get_action(sliding_window_frames, frame_counter):
    global nearest_idx, follow_frame, log

    if frame_counter < 20:  # Warmup phase ; Turn around
        action = env.action_space.noop()
        action['camera'] = [0, 10]
        return action

    follow_frame += 1
    if frame_counter % 20 == 0:  # Redo the search every N frames
        nearest_idx = search(sliding_window_frames)
        follow_frame = 0

    action, is_null_action = episode_actions.actions[nearest_idx + follow_frame]
    if action is None:
        print(f'[Frame {frame_counter} ({frame_counter // 20 // 60}:{(frame_counter // 20) % 60})] End of episode')
        nearest_idx = search(sliding_window_frames)
        return env.action_space.noop()

    return action


#env = gym.make('MineRLBasaltFindCave-v0')
env = HumanSurvival(**ENV_KWARGS).make()

obs = env.reset()

with torch.no_grad():
    for _ in tqdm(range(MAX_FRAMES)):

        frame = obs['pov']
        frame = np.transpose(cv2.resize(frame, AGENT_RESOLUTION), (2, 1, 0))
        sliding_window_frames.append(frame)
        if len(sliding_window_frames) > SLIDING_WINDOW_SIZE:
            sliding_window_frames.pop(0)

        action = get_action(sliding_window_frames, frame_counter)

        obs, reward, done, _ = env.step(action)  # obs['pov'].shape == (360, 640, 3)

        video_writer.write(cv2.cvtColor(obs['pov'], cv2.COLOR_RGB2BGR))

        frame_counter += 1
        env.render()

video_writer.release()
with open('agent_log.txt', 'w') as f:
    for m in log:
        f.write(m)
        f.write('\n')