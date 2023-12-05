import cv2
import torch
import minerl
import numpy as np
import torch.nn.functional as F

from EpisodeActions import EpisodeActions
from LatentSpaceMineCLIP import LatentSpaceMineCLIP, load_mineclip, AGENT_RESOLUTION, SLIDING_WINDOW_SIZE

class TargetedSearchAgent():
    def __init__(self, env, max_follow_frames=20, goal_rolling_window_size=20*20, device='cuda'):
        self.env = env
        self.past_frames = []
        self.frame_counter = 0  # How many frames the agent has played
        self.follow_frame = -1  # How many frames from the trajectory we have followed
        self.max_follow_frames = max_follow_frames  # How many frames we can follow before searching for a new trajectory

        self.redo_search_counter = 0  # TODO better name?
        self.redo_search_threshold = 5
        self.diff_threshold = 150.0  # TODO which number?

        self.diff_log = []
        self.search_log = []
        self.device = device

        self.mineclip_model = load_mineclip(device=self.device)
        self.episode_actions = EpisodeActions().load()
        self.latent_space_mineclip = LatentSpaceMineCLIP(device=self.device).load()

        self.nearest_idx = None
        self.current_goal = None
        self.future_goal_distances = None
        self.goal_rolling_window_size = goal_rolling_window_size
    
    def set_goal(self, goal_text):
        self.current_goal = goal_text
        text_latent = self.mineclip_model.encode_text(self.current_goal)[0].detach()
        goal_distances = self.latent_space_mineclip.get_distances(text_latent)

        goal_distances_padding = F.pad(goal_distances, (0, self.goal_rolling_window_size-1), 'constant', float('inf'))
        goal_distances_rolling = goal_distances_padding.unfold(0, self.goal_rolling_window_size, 1)
        self.future_goal_distances = goal_distances_rolling.min(1).values

        print(f'Set new goal: \"{self.current_goal}\"')

    def get_action(self, obs):
        self.follow_frame += 1
        self.frame_counter += 1

        latent = self.get_latent(obs)

        if self.frame_counter < 20:  # Warmup phase ; Turn around
            action = self.env.action_space.noop()
            action['camera'] = [0, 20]
            self.diff_log.append(0)  # TODO remove
            return action

        if self.should_search_again(latent):
            self.search(latent)

        action, is_null_action = self.episode_actions.actions[self.nearest_idx + self.follow_frame]

        return action
    
    def get_latent(self, obs):
        frame = obs['pov']
        frame = np.transpose(cv2.resize(frame, AGENT_RESOLUTION), (2, 1, 0))
        self.past_frames.append(frame)
        if len(self.past_frames) > SLIDING_WINDOW_SIZE:
            self.past_frames.pop(0)

        frame_window = torch.from_numpy(np.array(self.past_frames)).unsqueeze(0).to(self.device)
        latent = self.mineclip_model.encode_video(frame_window)[0]
        del(frame_window)

        return latent
    
    def search(self, latent):
        if self.current_goal is None:
            raise Exception('Goal is not set.') 

        possible_trajectories = self.future_goal_distances + self.latent_space_mineclip.get_distances(latent)
        self.nearest_idx = possible_trajectories.argmin().to('cpu').item()

        self.follow_frame = -1
        self.redo_search_counter = 0

        self.log_new_nearest()
    
    def should_search_again(self, latent):
        self.calc_follow_difference(latent)
        
        return self.nearest_idx is None \
            or self.follow_frame > self.max_follow_frames \
            or self.episode_actions.is_last(self.nearest_idx + self.follow_frame) \
            or self.redo_search_counter >= self.redo_search_threshold
        
    def calc_follow_difference(self, latent):
        if self.nearest_idx is None:
            self.diff_log.append(0)
            return

        diff_to_follow_latent = self.latent_space_mineclip.get_distance(self.nearest_idx + self.follow_frame, latent)
        self.diff_log.append(diff_to_follow_latent.to('cpu').item())

        if self.follow_frame < 0.33 * self.max_follow_frames:
            return  # TODO put back in prior `if`

        if diff_to_follow_latent > self.diff_threshold:
            self.redo_search_counter += 1
        else:
            self.redo_search_counter = min(self.redo_search_counter - 1, 0)

    def log_new_nearest(self):
        episode = 0
        while episode+1 < len(self.episode_actions.episode_starts) and int(self.episode_actions.episode_starts[episode+1][1]) <= self.nearest_idx:  # TODO remove int()
            episode += 1
        episode_id, episode_start = self.episode_actions.episode_starts[episode]
        episode_start = int(episode_start)

        episode_frame = self.nearest_idx - episode_start + SLIDING_WINDOW_SIZE - 1
        self.search_log.append(f'[Frame {self.frame_counter} ({self.frame_counter // 20 // 60}:{(self.frame_counter // 20) % 60})] Found nearest in {episode_id} at frame {episode_frame} ({episode_frame // 20 // 60}:{(episode_frame // 20) % 60})')
