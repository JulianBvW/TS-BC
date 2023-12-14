import cv2
import torch
import numpy as np
from tqdm import tqdm

from VPTDataset import VPTDataset
from LatentSpaceVPT import load_vpt, CONTEXT, AGENT_RESOLUTION

VID_ID = 'data/10.0/' + 'squeaky-magnolia-ocelot-bf12350083f0-20220418-182741'

def draw_single_dim(frame, idx, val):
    c_strength = int(abs(val)/100*255)
    color = (c_strength, 255-c_strength, 0)
    start, end = (5+idx*2, 5+100-abs(val)), (5+idx*2+2, 5+100)
    if val < 0:
        start, end = (start[0], start[1]+2+abs(val)), (end[0], end[1]+2+abs(val))

    frame = cv2.rectangle(frame, start, end, color, -1)
    return frame

vpt_model = load_vpt()
model_state = vpt_model.initial_state(1)
dataset = VPTDataset()
video_original, _, _ = dataset.get_from_vid_id(VID_ID)
video_original = video_original[:2000]

# Resize the video
resized_frames = np.empty((video_original.shape[0], AGENT_RESOLUTION[1], AGENT_RESOLUTION[0], 3), dtype=np.uint8)
for ts in range(video_original.shape[0]):
    resized_frame = cv2.resize(video_original[ts], AGENT_RESOLUTION)
    resized_frames[ts] = resized_frame
video = torch.tensor(resized_frames, requires_grad=False).to('cuda')

# Compute the latent differences
print('# Compute diffs...')
latent_diffs = []
last_latent = None
with torch.no_grad():
    for ts in tqdm(range(len(video))):
        frame = video[ts].unsqueeze(0).unsqueeze(0)
        (latent, _), model_state = vpt_model.net({'img': frame}, model_state, context=CONTEXT)
        latent = latent[0][0].to('cpu').numpy().astype('float16')
        if last_latent is None:
            last_latent = latent
        latent_diffs.append(latent - last_latent)
        last_latent = latent

latent_diffs = np.array(latent_diffs)
max_val = abs(latent_diffs).max()
latent_diffs = (latent_diffs/max_val*100).astype(int)

# Draw latent diffs
print('# Draw diffs...')
analysis_video = np.empty((video.shape[0], 200+2+10, 1024*2+10, 3), dtype=np.uint8) + 255
for ts in tqdm(range(analysis_video.shape[0])):
    analysis_video[ts] = cv2.rectangle(analysis_video[ts], (5, 100+5), (1024*2+5, 100+5+2), (0, 0, 0), -1)
    for latent_idx in range(latent_diffs.shape[1]):
        analysis_video[ts] = draw_single_dim(analysis_video[ts], latent_idx, latent_diffs[ts][latent_idx])

# Combine videos
VIDEO_RESOLUTION = (1024*2+10, 1158)
resized_frames = np.empty((video_original.shape[0], VIDEO_RESOLUTION[1], VIDEO_RESOLUTION[0], 3), dtype=np.uint8)
for ts in range(video_original.shape[0]):
    resized_frame = cv2.resize(video_original[ts], VIDEO_RESOLUTION)
    resized_frames[ts] = resized_frame
analysis_video = np.concatenate([resized_frames, analysis_video], axis=1)

# Save video
video_writer = cv2.VideoWriter('output/vpt_latent_analysis.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20, (analysis_video.shape[2], analysis_video.shape[1]))
for frame in analysis_video:
    video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
video_writer.release()


