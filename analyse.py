import cv2
import numpy as np
import skvideo.io
from tqdm import tqdm

from VPTDataset import VPTDataset

# Load agent recording
agent_recording = skvideo.io.vread('output/agent_recording.mp4')
agent_duration = agent_recording.shape[0]

# Read duration of followed trajectory
dataset = VPTDataset()
with open('output/agent_log.txt', 'r') as f:
    log = f.readlines()
    agent_frames = list(map(lambda x: int(x.split(' ')[1]), log))
    vid_ids      = list(map(lambda x: str(x.split(' ')[6]), log))
    vid_frames   = list(map(lambda x: int(x.split(' ')[9]), log))
with open('output/agent_diff_log.txt', 'r') as f:
    diff_log = f.readlines()
    diffs = list(map(lambda x: float(x), diff_log))
    mn, mx = min(diffs[22:]), max(diffs)
durations = [(agent_frames+[agent_duration])[i+1] - agent_frames[i] for i in range(len(agent_frames))]

# Create Dataset video
dataset_video = np.empty((agent_duration, 360, 640, 3), dtype=np.uint8)
for agent_frame, vid_id, vid_frame, duration in tqdm(list(zip(agent_frames, vid_ids, vid_frames, durations))):
    vid, _, _ = dataset.get_from_vid_id(vid_id)
    dataset_video[agent_frame:agent_frame+duration] = vid[vid_frame:vid_frame+duration]

for i in range(len(dataset_video)):
    dataset_video[i] = cv2.rectangle(dataset_video[i], (0, 0), (110, 32), (255, 255, 255), -1)
    color_ramp = max(0, int((diffs[i]-mn)/(mx-mn)*255))
    dataset_video[i] = cv2.putText(dataset_video[i], f'{round(diffs[i], 1): 6}', (0, 27), cv2.FONT_HERSHEY_SIMPLEX, 1, (color_ramp, 255-color_ramp, 0), 2, cv2.LINE_AA)

comb_video = np.concatenate([agent_recording, dataset_video], axis=1)

# Write mp4 file
video_writer = cv2.VideoWriter('output/analysis_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20, (640, 2*360))
for frame in comb_video:
    video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
video_writer.release()
