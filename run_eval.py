import os
import argparse
from datetime import datetime

from run_agent import main as run_agent_main
from distance_fns import DISTANCE_FUNCTIONS

GOALS = ['dig as far as possible', 'get dirt', 'look at the sky', 'break leaves', 'chop a tree', 'collect seeds', 'break a flower', 'go explore', 'go swimming', 'go underwater', 'open inventory']

def main(args):
    name = args.name or datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    base_dir = 'eval/' + name
    os.makedirs(f'{base_dir}/', exist_ok=True)

    # Save configuration
    with open(f'{base_dir}/values.txt', 'w') as f:
        for k, v in vars(args).items():
            f.write(f'{k}: {v}\n')
    
    # Run agent for each goal multiple times
    for goal in GOALS:
        for i in range(args.runs_per_goal):
            args.goal = goal
            args.output_dir = base_dir + '/' + goal.replace(' ', '_') + f'/{i}'
            run_agent_main(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--runs-per-goal', type=int, default=10)
    parser.add_argument('--max-frames', type=int, default=1*60*20)

    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--description', type=str, default='')
    parser.add_argument('--distance-fn', type=str, default='cosine', choices=DISTANCE_FUNCTIONS.keys())

    args = parser.parse_args()

    main(args)