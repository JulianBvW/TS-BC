import cv2
import gym
import torch
import minerl
import numpy as np
from tqdm import tqdm
from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival

from util import to_minerl_action
from EpisodeActions import EpisodeActions
from LatentSpaceMineCLIP import SLIDING_WINDOW_SIZE
from LatentSpaceVPT import LatentSpaceVPT, load_vpt, AGENT_RESOLUTION, CONTEXT

MAX_FRAMES = 2*60*20
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

vpt_model = load_vpt()
model_state = vpt_model.initial_state(1)
follow_frame = 0
log = []

episode_actions = EpisodeActions()
episode_actions.load()

latent_space_vpt = LatentSpaceVPT()
latent_space_vpt.load()

def search(frame):
    global model_state, log, frame_counter

    frame = cv2.resize(frame, AGENT_RESOLUTION)
    frame = torch.tensor(frame).unsqueeze(0).unsqueeze(0).to('cuda')  # Add 2 extra dimensions for vpt
    (latent, _), model_state = vpt_model.net({'img': frame}, model_state, context=CONTEXT)
    del(frame)

    nearest_idx = latent_space_vpt.get_nearest(latent[0][0])
    nearest_idx = nearest_idx.to('cpu').item()

    episode = 0
    while episode+1 < len(episode_actions.episode_starts) and int(episode_actions.episode_starts[episode+1][1]) <= nearest_idx:
        episode += 1
    episode_id, episode_start = episode_actions.episode_starts[episode]
    episode_start = int(episode_start)

    episode_frame = nearest_idx - episode_start + SLIDING_WINDOW_SIZE #-/+ 1 # TODO
    log.append(f'[Frame {frame_counter} ({frame_counter // 20 // 60}:{(frame_counter // 20) % 60})] Found nearest in {episode_id} at frame {episode_frame} ({episode_frame // 20 // 60}:{(episode_frame // 20) % 60})')

    return nearest_idx

def get_action(frame, frame_counter):
    global nearest_idx, follow_frame, log

    if frame_counter < 40:  # Warmup phase ; Turn around
        action = env.action_space.noop()
        action['camera'] = [0, 10]
        return action

    follow_frame += 1
    if frame_counter == 40 or frame_counter % 60 == 0:  # Redo the search every N frames
        nearest_idx = search(frame)
        follow_frame = 0

    action = episode_actions.actions[nearest_idx + follow_frame]
    if action is None:
        log.append(f'[Frame {frame_counter} ({frame_counter // 20 // 60}:{(frame_counter // 20) % 60})] End of episode')
        nearest_idx = search(frame)
        return env.action_space.noop()

    action, _ = to_minerl_action(action)  # TODO directly at training

    return action


#env = gym.make('MineRLBasaltFindCave-v0')
env = HumanSurvival(**ENV_KWARGS).make()

obs = env.reset()

with torch.no_grad():
    for _ in tqdm(range(MAX_FRAMES)):
        action = get_action(obs['pov'], frame_counter)

        obs, reward, done, _ = env.step(action)  # obs['pov'].shape == (360, 640, 3)

        video_writer.write(cv2.cvtColor(obs['pov'], cv2.COLOR_RGB2BGR))

        frame_counter += 1
        env.render()

video_writer.release()
with open('agent_log.txt', 'w') as f:
    for m in log:
        f.write(m)
        f.write('\n')