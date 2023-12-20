import cv2
import torch
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from mineclip import MineCLIP

from distance_fns import DISTANCE_FUNCTIONS

AGENT_RESOLUTION = (256, 160)  # (W, H)
SLIDING_WINDOW_SIZE = 16

class LatentSpaceMineCLIP:
    def __init__(self, distance_fn='euclidean', device='cuda'):
        self.latents = []  # Python List while training, Numpy array while inference
        self.distance_function = DISTANCE_FUNCTIONS[distance_fn]
        self.device = device
    
    @torch.no_grad()
    def load(self, episode_actions, latents_folder='weights/ts_bc/latents_mineclip/'):
        for vid_id, _ in episode_actions.episode_starts:
            _, name = vid_id.rsplit('/', 1)
            vid_latents = np.load(latents_folder + name + '.npy', allow_pickle=True)
            self.latents.append(vid_latents)

        self.latents = torch.from_numpy(np.vstack(self.latents)).to(self.device)
        print(f'Loaded MineCLIP latent space with {len(self.latents)} latents')
        return self
    
    @torch.no_grad()
    def load_OLD(self, latents_file='weights/ts_bc/latents_mineclip.npy'):  # TODO update to new format
        self.latents = torch.from_numpy(np.load(latents_file, allow_pickle=True)).to(self.device)
        print(f'Loaded MineCLIP latent space with {len(self.latents)} latents')
        return self
    
    def save(self, latents_file='weights/ts_bc/latents_mineclip'):  # TODO remove?
        latents = np.array(self.latents)
        np.save(latents_file, latents)

    @torch.no_grad()
    def train_episode(self, mineclip_model, frames, vid_id, save_dir='weights/ts_bc/latents_mineclip/'):
        episode_latents = []

        resized_frames = np.empty((frames.shape[0], AGENT_RESOLUTION[1], AGENT_RESOLUTION[0], 3), dtype=np.uint8)
        for ts in range(frames.shape[0]):
            resized_frame = cv2.resize(frames[ts], AGENT_RESOLUTION)
            resized_frames[ts] = resized_frame

        sliding_window_frames = sliding_window_view(resized_frames, SLIDING_WINDOW_SIZE, 0)
        sliding_window_frames = torch.tensor(np.transpose(sliding_window_frames, (0, 4, 3, 2, 1)))

        inter_batch_size = 100
        for i in range(sliding_window_frames.shape[0] // inter_batch_size + 1):
            inter_batch_frames = sliding_window_frames[i*inter_batch_size:(i+1)*inter_batch_size].to(self.device)

            latents = mineclip_model.encode_video(inter_batch_frames)
            
            episode_latents += [latent.astype('float16') for latent in latents.to('cpu').numpy()]

            del(inter_batch_frames)
        
        np.save(save_dir + vid_id.rsplit('/', 1)[-1], np.array(episode_latents))

    def get_distances(self, latent):
        return self.distance_function(self.latents, latent)
    
    def get_distance(self, idx, latent):
        return self.distance_function(self.latents[idx], latent)

    def get_nearest(self, latent): # TODO episode_start is removed
        diffs = self.get_distances(latent)
        nearest_idx = diffs.argmin()#.to('cpu').item() # TODO remove .to('cpu').item()
        return nearest_idx

def load_mineclip(weights_file='weights/mineclip/attn.pth', device='cuda'):  # TODO: in it's own file?
    mineclip = MineCLIP(arch='vit_base_p16_fz.v2.t2', hidden_dim=512, image_feature_dim=512, mlp_adapter_spec='v0-2.t0', pool_type='attn.d2.nh8.glusw', resolution=[160, 256])
    mineclip.load_ckpt(weights_file, strict=True)
    mineclip.eval()

    return mineclip.to(device)
