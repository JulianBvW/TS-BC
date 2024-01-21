import os
import argparse
from tqdm import tqdm

from VPTDataset import VPTDataset
from EpisodeActions import EpisodeActions
from LatentSpaceVPT import LatentSpaceVPT, load_vpt
from LatentSpaceMineCLIP import LatentSpaceMineCLIP, load_mineclip

def main(args):
    print('Computing Latent Vectors...')

    save_dir = args.save_dir
    os.makedirs(save_dir + '/actions/', exist_ok=True)
    os.makedirs(save_dir + '/latents_vpt/', exist_ok=True)
    os.makedirs(save_dir + '/latents_mineclip/', exist_ok=True)

    dataset = VPTDataset()

    episode_actions = EpisodeActions()
    latent_space_vpt = LatentSpaceVPT()
    latent_space_mineclip = LatentSpaceMineCLIP()

    vpt_model = load_vpt()
    mineclip_model = load_mineclip()

    iterator = range(args.batch_size) if args.random_sample_size is None else tqdm(range(args.random_sample_size))

    for i in tqdm(iterator):
        if args.random_sample_size is None:
            idx = args.batch_idx * args.batch_size + i
            if idx >= len(dataset):
                break
            frames, actions, vid_id = dataset[idx]
        else:
            frames, actions, vid_id = dataset.get_random()

        if frames is None:
            print(f'SKIPPING index: {idx}')
            continue

        episode_actions.train_episode(actions, vid_id, save_dir=save_dir + '/actions/')
        latent_space_vpt.train_episode(vpt_model, frames, vid_id, save_dir=save_dir + '/latents_vpt/')
        latent_space_mineclip.train_episode(mineclip_model, frames, vid_id, save_dir=save_dir + '/latents_mineclip/')

        dataset.delete(vid_id)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--random-sample-size', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=400)
    parser.add_argument('--batch-idx', type=int, default=0)
    parser.add_argument('--save-dir', type=str, default='weights/ts_bc/')
    args = parser.parse_args()

    main(args)
