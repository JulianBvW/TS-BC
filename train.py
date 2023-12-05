import argparse
from tqdm import tqdm

from VPTDataset import VPTDataset
from EpisodeActions import EpisodeActions
from LatentSpaceVPT import LatentSpaceVPT, load_vpt
from LatentSpaceMineCLIP import LatentSpaceMineCLIP, load_mineclip

def main(args):
    dataset = VPTDataset()

    episode_actions = EpisodeActions()
    latent_space_vpt = LatentSpaceVPT()
    latent_space_mineclip = LatentSpaceMineCLIP()

    vpt_model = load_vpt()
    mineclip_model = load_mineclip()

    for i in tqdm(range(args.sample_size)):
        frames, actions, vid_id = dataset.get_random_and_delete()  # TODO not the same on multiple times

        episode_actions.train_episode(actions, vid_id)
        latent_space_vpt.train_episode(vpt_model, frames)
        latent_space_mineclip.train_episode(mineclip_model, frames)

        if i % 20 == 0:
            episode_actions.save()
            latent_space_vpt.save()
            latent_space_mineclip.save()
    
    episode_actions.save()
    latent_space_vpt.save()
    latent_space_mineclip.save()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--sample-size', type=int, default=200)
    args = parser.parse_args()

    main(args)