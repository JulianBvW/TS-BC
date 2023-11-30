import cv2
import torch
import numpy as np
from PIL import Image
from LatentSpaceVPT import LatentSpaceVPT, load_vpt

vpt = load_vpt()

plains_tree_less = cv2.resize(np.array(Image.open('example_images/plains_tree_less.png'))[:, :, :3], (128, 128))
plains_tree      = cv2.resize(np.array(Image.open('example_images/plains_tree.png'     ))[:, :, :3], (128, 128))
desert_tree      = cv2.resize(np.array(Image.open('example_images/desert_tree.png'     ))[:, :, :3], (128, 128))
plains           = cv2.resize(np.array(Image.open('example_images/plains.png'          ))[:, :, :3], (128, 128))
desert           = cv2.resize(np.array(Image.open('example_images/desert.png'          ))[:, :, :3], (128, 128))

def get_after_1_sec(img):
    state = vpt.initial_state(1)
    img = {'img': torch.tensor(img).unsqueeze(0).unsqueeze(0).to('cuda')}
    for _ in range(100):
        (latent, _), state = vpt.net(img, state, context={'first': torch.tensor([[False]])})
    return latent

with torch.no_grad():
    lptl = get_after_1_sec(plains_tree_less)
    lpt = get_after_1_sec(plains_tree)
    ldt = get_after_1_sec(desert_tree)
    lp = get_after_1_sec(plains)
    ld = get_after_1_sec(desert)

vpt_latent_space = LatentSpaceVPT()
vpt_latent_space.load()

print('### plains_tree_less')
vpt_latent_space.get_nearest(lptl)
print('### plains_tree')
vpt_latent_space.get_nearest(lpt)
print('### desert_tree')
vpt_latent_space.get_nearest(ldt)
print('### plains')
vpt_latent_space.get_nearest(lp)
print('### desert')
vpt_latent_space.get_nearest(ld)