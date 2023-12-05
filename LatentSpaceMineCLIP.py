import cv2
import torch
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from mineclip import MineCLIP

AGENT_RESOLUTION = (256, 160)  # (W, H)
SLIDING_WINDOW_SIZE = 16

class LatentSpaceMineCLIP:
    def __init__(self):
        self.latents = []  # Python List while training, Numpy array while inference
    
    @torch.no_grad()
    def load(self, latents_file='weights/ts_bc/latents_mineclip.npy'):
        self.latents = torch.from_numpy(np.load(latents_file, allow_pickle=True)).to('cuda')
        print(f'Loaded MineCLIP latent space with {len(self.latents)} latents')
        return self
    
    def save(self, latents_file='weights/ts_bc/latents_mineclip'):
        latents = np.array(self.latents)
        np.save(latents_file, latents)

    @torch.no_grad()
    def train_episode(self, mineclip_model, frames):

        resized_frames = np.empty((frames.shape[0], AGENT_RESOLUTION[1], AGENT_RESOLUTION[0], 3), dtype=np.uint8)
        for ts in range(frames.shape[0]):
            resized_frame = cv2.resize(frames[ts], AGENT_RESOLUTION)
            resized_frames[ts] = resized_frame

        sliding_window_frames = sliding_window_view(resized_frames, SLIDING_WINDOW_SIZE, 0)
        sliding_window_frames = torch.tensor(np.transpose(sliding_window_frames, (0, 4, 3, 2, 1)))

        inter_batch_size = 100
        for i in range(sliding_window_frames.shape[0] // inter_batch_size + 1):
            inter_batch_frames = sliding_window_frames[i*inter_batch_size:(i+1)*inter_batch_size].to('cuda')

            latents = mineclip_model.encode_video(inter_batch_frames)
            
            self.latents += [latent.astype('float16') for latent in latents.to('cpu').numpy()]

            del(inter_batch_frames)

        #frames = torch.tensor(np.transpose(resized_frames, (0, 3, 2, 1))).to('cuda') # TODO Batch Training with: np.transpose(sliding_window_view(frames, 5, 0), (0, 4, 1, 2, 3))

        # for ts in range(16, len(frames)):
        #     frame_window = frames[ts-16:ts].unsqueeze(0)  # Add 1 extra dimension for mineclip
        #     frame_latents = mineclip_model.forward_image_features(frame_window)
        #     latent = mineclip_model.forward_video_features(frame_latents)
        #     latent = latent[0].to('cpu').numpy().astype('float16')

        #     self.latents.append(latent)

        # del(frames)

    def get_distances(self, latent):  # TODO look at different norms
        diffs = self.latents - latent
        diffs = abs(diffs).sum(1)  # Sum up along the single latents exponential difference to the current latent
        return diffs
    
    def get_distance(self, idx, latent):  # TODO refactor to use `self.distance_function` or something
        diff = self.latents[idx] - latent
        diff = abs(diff).sum()
        return diff

    def get_nearest(self, latent): # TODO episode_start is removed
        diffs = self.get_distances(latent)
        nearest_idx = diffs.argmin()#.to('cpu').item() # TODO remove .to('cpu').item()
        return nearest_idx

        episode = 0
        while episode+1 < len(self.episode_starts) and int(self.episode_starts[episode+1][1]) <= nearest_idx:
            episode += 1
        episode_id, episode_start = self.episode_starts[episode]
        episode_start = int(episode_start)

        print(f'Found nearest in {episode_id} at frame {nearest_idx - episode_start} ({(nearest_idx - episode_start) // 20 // 60}:{((nearest_idx - episode_start) // 20) % 60})')

def load_mineclip(weights_file='weights/mineclip/mineclip-attn.pth'):  # TODO: in it's own file?
    mineclip = MineCLIP(arch='vit_base_p16_fz.v2.t2', hidden_dim=512, image_feature_dim=512, mlp_adapter_spec='v0-2.t0', pool_type='attn.d2.nh8.glusw', resolution=[160, 256])
    mineclip.load_ckpt(weights_file, strict=True)

    return mineclip.to('cuda')
