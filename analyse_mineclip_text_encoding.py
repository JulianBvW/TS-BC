import cv2
import torch
import numpy as np
from tqdm import tqdm

from LatentSpaceMineCLIP import LatentSpaceMineCLIP, load_mineclip, SLIDING_WINDOW_SIZE
from EpisodeActions import EpisodeActions
from VPTDataset import VPTDataset
from CVAE import load_cvae

cvae_model = load_cvae()
mineclip_model = load_mineclip()
latent_space_mineclip = LatentSpaceMineCLIP().load()
episode_actions = EpisodeActions().load()
dataset = VPTDataset()

VID_DIMS = (64*4, 36*4)  # (256, 144)
#GOALS = ['chop a tree', 'collect dirt', 'kill a pig', 'build a tower', 'collect seeds', 'break tall grass', 'go explore']
#GOALS = ['collect dirt', 'get dirt, dig hole, dig dirt, gather a ton of dirt, collect dirt']
GOALS = ['break tall grass', 'break tall grass, break grass, collect seeds, punch the ground, run around in circles getting seeds from bushes']

vids = {}

for goal in tqdm(GOALS):
    #print(f'### {goal}')
    vids[goal] = []
    text_latent = mineclip_model.encode_text(goal).detach()
    vis_text_latent = cvae_model(text_latent)
    #goal_distances = latent_space_mineclip.get_distances(text_latent[0])
    goal_distances = latent_space_mineclip.get_distances(vis_text_latent[0])
    for i in range(9):
        nearest_idx = goal_distances.argmin().to('cpu').item()
        #nearest_idx = torch.abs((goal_distances - torch.quantile(goal_distances, 0.1))).argmin()

        episode = 0
        while episode+1 < len(episode_actions.episode_starts) and int(episode_actions.episode_starts[episode+1][1]) <= nearest_idx:
            episode += 1
        episode_id, episode_start = episode_actions.episode_starts[episode]
        episode_start = int(episode_start)

        episode_frame = nearest_idx - episode_start + SLIDING_WINDOW_SIZE - 1
        episode_url = 'https://openaipublic.blob.core.windows.net/minecraft-rl/' + episode_id + '.mp4'
        #print(f'{i}. ({episode_frame // 20 // 60}:{(episode_frame // 20) % 60:02}) in {episode_url}')
        vids[goal].append([episode_id, episode_frame])

        goal_distances[max(nearest_idx-200, 0):nearest_idx+200] += 100.0


print('Creating video..')

video_writer = None

for goal in GOALS:
    print(f'### {goal}')

    bestvids = []
    for i in tqdm(range(9)):
        vid_id, vid_frame = vids[goal][i]
        vid, _, _ = dataset.get_from_vid_id(vid_id)
        vid_resize = []
        for frame in vid[vid_frame-10:vid_frame+10]:
            vid_resize.append(cv2.resize(frame, VID_DIMS))
        vid_resize = np.array(vid_resize)
        bestvids.append(vid_resize)
    comb_video_top = np.concatenate(bestvids[:3], axis=2)
    comb_video_mid = np.concatenate(bestvids[3:6], axis=2)
    comb_video_bot = np.concatenate(bestvids[6:], axis=2)
    comb_video = np.concatenate([comb_video_top, comb_video_mid, comb_video_bot], axis=1)
    for i in range(len(comb_video)):
        comb_video[i] = cv2.putText(comb_video[i], goal, (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    
    if video_writer is None:
        video_writer = cv2.VideoWriter('output/mineclip_text_analysis.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20, (comb_video.shape[2], comb_video.shape[1]))

    # Write mp4 file
    for _ in range(5):
        for frame in comb_video:
            video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))




video_writer.release()
