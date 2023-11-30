import cv2
import torch
import numpy as np
from PIL import Image
from LatentSpaceVPT import load_vpt

vpt = load_vpt()

plains_tree_less = cv2.resize(np.array(Image.open('example_images/plains_tree_less.png'))[:, :, :3], (128, 128))
plains_tree      = cv2.resize(np.array(Image.open('example_images/plains_tree.png'     ))[:, :, :3], (128, 128))
desert_tree      = cv2.resize(np.array(Image.open('example_images/desert_tree.png'     ))[:, :, :3], (128, 128))
plains           = cv2.resize(np.array(Image.open('example_images/plains.png'          ))[:, :, :3], (128, 128))
desert           = cv2.resize(np.array(Image.open('example_images/desert.png'          ))[:, :, :3], (128, 128))

with torch.no_grad():
    (lptl, _), _ = vpt.net({'img': torch.tensor(plains_tree_less).unsqueeze(0).unsqueeze(0).to('cuda')}, vpt.initial_state(1), context={'first': torch.tensor([[False]])})
    (lpt,  _), _ = vpt.net({'img': torch.tensor(plains_tree     ).unsqueeze(0).unsqueeze(0).to('cuda')}, vpt.initial_state(1), context={'first': torch.tensor([[False]])})
    (ldt,  _), _ = vpt.net({'img': torch.tensor(desert_tree     ).unsqueeze(0).unsqueeze(0).to('cuda')}, vpt.initial_state(1), context={'first': torch.tensor([[False]])})
    (lp,   _), _ = vpt.net({'img': torch.tensor(plains          ).unsqueeze(0).unsqueeze(0).to('cuda')}, vpt.initial_state(1), context={'first': torch.tensor([[False]])})
    (ld,   _), _ = vpt.net({'img': torch.tensor(desert          ).unsqueeze(0).unsqueeze(0).to('cuda')}, vpt.initial_state(1), context={'first': torch.tensor([[False]])})

def diff(a, b):
    return ((a-b)**2).sum().sqrt().to('cpu').item()



