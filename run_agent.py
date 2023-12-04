import os
import cv2
import torch
import minerl
from tqdm import tqdm
from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival

from TargetedSearchAgent import TargetedSearchAgent

TEXT_GOAL = 'gather wood'

MAX_FRAMES = 1*60*20
ENV_KWARGS = dict(
    fov_range=[70, 70],
    frameskip=1,
    gamma_range=[2, 2],
    guiscale_range=[1, 1],
    resolution=[640, 360],
    cursor_size_range=[16.0, 16.0],
)

video_writer = cv2.VideoWriter('output/agent_recording.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20, (640, 360))
os.makedirs('output/', exist_ok=True)

env = HumanSurvival(**ENV_KWARGS).make()
agent = TargetedSearchAgent(env)
agent.set_goal(TEXT_GOAL)

obs = env.reset()
with torch.no_grad():
    for _ in tqdm(range(MAX_FRAMES)):

        action = agent.get_action(obs)
        obs, reward, done, _ = env.step(action)

        video_writer.write(cv2.cvtColor(obs['pov'], cv2.COLOR_RGB2BGR))
        env.render()

video_writer.release()
with open('output/agent_log.txt', 'w') as f:
    for m in agent.log:
        f.write(m + '\n')