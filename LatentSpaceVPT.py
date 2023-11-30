import sys
sys.path.append('openai_vpt')

import cv2
import torch
import pickle
import numpy as np
from gym3.types import DictType
from openai_vpt.lib.policy import MinecraftAgentPolicy

from LatentSpaceMineCLIP import SLIDING_WINDOW_SIZE

AGENT_RESOLUTION = (128, 128)
CONTEXT = {'first': torch.tensor([[False]])}

class LatentSpaceVPT:
    def __init__(self):
        self.latents = []  # Python List while training, Numpy array while inference
    
    @torch.no_grad()
    def load(self, latents_file='weights/ts_bc/latents_vpt.npy'):
        self.latents = torch.from_numpy(np.load(latents_file, allow_pickle=True)).to('cuda')
        print(f'Loaded VPT latent space with {len(self.latents)} latents')
    
    def save(self, latents_file='weights/ts_bc/latents_vpt'):
        latents = np.array(self.latents)
        np.save(latents_file, latents)

    @torch.no_grad()
    def train_episode(self, vpt_model, frames):
        model_state = vpt_model.initial_state(1)

        resized_frames = np.empty((frames.shape[0], AGENT_RESOLUTION[1], AGENT_RESOLUTION[0], 3), dtype=np.uint8)
        for ts in range(frames.shape[0]):
            resized_frame = cv2.resize(frames[ts], AGENT_RESOLUTION)
            resized_frames[ts] = resized_frame
        frames = torch.tensor(resized_frames).to('cuda')

        for ts in range(SLIDING_WINDOW_SIZE-1, len(frames)):  # Start at Frame 15 because of MineCLIP needing 16-frame batches
            frame = frames[ts].unsqueeze(0).unsqueeze(0)  # Add 2 extra dimensions for vpt
            (latent, _), model_state = vpt_model.net({'img': frame}, model_state, context=CONTEXT)
            latent = latent[0][0].to('cpu').numpy().astype('float16')

            self.latents.append(latent)
        
        del(frames)

    def get_nearest(self, latent): # TODO removed episode_starts
        # TODO assert latents is numpy array

        diffs = self.latents - latent
        diffs = (diffs**2).sum(1)  # Sum up along the single latents exponential difference to the current latent
        nearest_idx = diffs.argmin()#.to('cpu').item() # TODO remove .to('cpu').item()
        return nearest_idx

        episode = 0
        while episode+1 < len(self.episode_starts) and int(self.episode_starts[episode+1][1]) <= nearest_idx:
            episode += 1
        episode_id, episode_start = self.episode_starts[episode]
        episode_start = int(episode_start)

        print(f'Found nearest in {episode_id} at frame {nearest_idx - episode_start} ({(nearest_idx - episode_start) // 20 // 60}:{((nearest_idx - episode_start) // 20) % 60})')

        return nearest_idx

def load_vpt(model_file='weights/vpt/foundation-model-1x.model', weights_file='weights/vpt/foundation-model-1x.weights'):
    agent_parameters = pickle.load(open(model_file, 'rb'))

    policy_kwargs = agent_parameters['model']['args']['net']['args']
    pi_head_kwargs = agent_parameters['model']['args']['pi_head_opts']
    pi_head_kwargs['temperature'] = float(pi_head_kwargs['temperature'])

    agent = MinecraftAgentPolicy(policy_kwargs=policy_kwargs, pi_head_kwargs=pi_head_kwargs, action_space=DictType())
    agent.load_state_dict(torch.load(weights_file), strict=False)

    return agent.to('cuda')
