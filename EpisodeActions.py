import torch
import numpy as np

from util import to_minerl_action
from LatentSpaceMineCLIP import SLIDING_WINDOW_SIZE

class EpisodeActions:
    def __init__(self):
        self.actions = []         # Python List while training, Numpy array while inference
        self.episode_starts = []  # Python List while training, Numpy array while inference

        self.frame_counter = 0

    @torch.no_grad()
    def load(self, actions_file='weights/ts_bc/actions.npy', episode_starts_file='weights/ts_bc/episode_starts.npy'):  # TODO new format
        self.actions = np.load(actions_file, allow_pickle=True)
        self.episode_starts = np.load(episode_starts_file, allow_pickle=False)

        print(f'Loaded actions from {len(self.episode_starts)} episodes')
        return self

    def save(self, actions_file='weights/ts_bc/actions', episode_starts_file='weights/ts_bc/episode_starts'):  # TODO remove
        actions = np.array(self.actions)
        episode_starts = np.array(self.episode_starts)
        
        np.save(actions_file, actions)
        np.save(episode_starts_file, episode_starts)

    @torch.no_grad()
    def train_episode(self, actions, vid_id, save_dir='weights/ts_bc/actions/'):
        episode_actions = []
        
        for ts in range(SLIDING_WINDOW_SIZE-1, len(actions)):  # Start at Frame 15 because of MineCLIP needing 16-frame batches
            episode_actions.append(to_minerl_action(actions[ts]))
        
        episode_actions.append(to_minerl_action(None))  # Append Null Action for last frame

        np.save(save_dir + vid_id.rsplit('/', 1)[-1], np.array(episode_actions))
    
    def is_last(self, idx):
        return str(idx + 1) in self.episode_starts[:, 1] or idx + 1 >= len(self.actions)