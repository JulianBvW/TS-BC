import os
import torch
import random
import numpy as np

from util import to_minerl_action
from LatentSpaceMineCLIP import SLIDING_WINDOW_SIZE

class EpisodeActions:
    def __init__(self):
        self.actions = []         # Python List while training, Numpy array while inference
        self.episode_starts = []  # Python List while training, Numpy array while inference

    @torch.no_grad()
    def load(self, search_folder='weights/ts_bc/', N=3):
        action_files = os.listdir(search_folder + 'actions/')
        latents_vpt_files = os.listdir(search_folder + 'latents_vpt')
        latents_mineclip_files = os.listdir(search_folder + 'latents_mineclip')

        print('Found:')
        print(f'   Action Files: {len(action_files)}')
        print(f'   VPT Files:    {len(latents_vpt_files)}')
        print(f'   MCLIP Files:  {len(latents_mineclip_files)}')

        # Filter out corrupted indexing files
        action_files = [af for af in action_files if af in latents_vpt_files and af in latents_mineclip_files]
        print(f'After filtering: {len(action_files)}')

        # Sample latents if wanted
        if N > 0:
            random.shuffle(action_files)
            action_files = action_files[:N]

        # Load actions
        frame_counter = 0
        for action_file in action_files:
            actions = np.load(search_folder + 'actions/' + action_file, allow_pickle=True)
            self.actions.append(actions)

            name = 'data/10.0/' + action_file.rsplit('.', 1)[0]
            self.episode_starts.append((name, frame_counter))

            frame_counter += actions.shape[0]

        self.actions = np.vstack(self.actions)
        self.episode_starts = np.array(self.episode_starts)
        print(f'Loaded actions from {len(self.episode_starts)} episodes')
        return self

    @torch.no_grad()
    def load_OLD(self, actions_file='weights/ts_bc/actions.npy', episode_starts_file='weights/ts_bc/episode_starts.npy'):  # TODO new format
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
