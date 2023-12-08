import os
import cv2
import torch
import minerl
import argparse
from tqdm import tqdm
from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival

from distance_fns import DISTANCE_FUNCTIONS
from TargetedSearchAgent import TargetedSearchAgent

ENV_KWARGS = dict(
    fov_range=[70, 70],
    frameskip=1,
    gamma_range=[2, 2],
    guiscale_range=[1, 1],
    resolution=[640, 360],
    cursor_size_range=[16.0, 16.0],
)

def main(args):
    video_writer = cv2.VideoWriter('output/agent_recording.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20, (640, 360))
    os.makedirs('output/', exist_ok=True)

    # HumanSurvival is a sandbox `gym`` environment for Minecraft with no set goal or timeframe
    env = HumanSurvival(**ENV_KWARGS).make()
    if args.seed is not None:
        env.seed(args.seed)

    agent = TargetedSearchAgent(env, distance_fn=args.distance_fn, device=args.device)
    agent.set_goal(args.goal)

    obs = env.reset()
    print('### Starting agent')
    with torch.no_grad():
        for _ in tqdm(range(args.max_frames)):

            action = agent.get_action(obs)
            obs, _, _, _ = env.step(action)

            video_writer.write(cv2.cvtColor(obs['pov'], cv2.COLOR_RGB2BGR))
            env.render()

    video_writer.release()
    with open('output/agent_log.txt', 'w') as f:
        for m in agent.search_log:
            f.write(m + '\n')
    with open('output/agent_diff_log.txt', 'w') as f:
        for m in agent.diff_log:
            f.write(str(m) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--distance_fn', type=str, default='euclidean', choices=DISTANCE_FUNCTIONS.keys())
    parser.add_argument('--goal', type=str, default='gather wood')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--max-frames', type=int, default=1*60*20)
    args = parser.parse_args()

    main(args)