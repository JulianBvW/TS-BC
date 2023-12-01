# TS-BC

**Targeted Search-Based Behavioral-Cloning** for learning to play Minecraft.

**Search-Based Behavioral-Cloning** (S-BC) searches the latent space of a neural model (here VPT) for similar situations in the dataset.
After finding a suitable starting point it copies the actions that were done in the demonstrator dataset.

**Targeted S-BC** (TS-BC) takes a text goal and searches the dataset for similar outcomes in another model (here MineCLIP).
The search results from both models are then filtered to get trajectories where the current agent situation can lead to the goal.
The actions from those trajectories are then copied.

### Installation

1. Make sure to have Java JDK 8 installed

2. Create a Conda environment
```
conda create --name ts-bc python=3.9
conda activate ts-bc
```
3. Install dependencies
```
pip install pip==21 setuptools==65.5.0 importlib-resources==1.3
pip install torch torchvision torchaudio opencv-python tqdm numpy==1.23.5 pandas scikit-video
pip install gym==0.21.0 gym3 attrs
```
4. Install [MineRL](https://github.com/minerllabs/minerl) and [MineCLIP](https://github.com/MineDojo/MineCLIP)
```
pip install git+https://github.com/minerllabs/minerl
pip install git+https://github.com/MineDojo/MineCLIP
```
5. Download weights and put them into `./weights/{vpt or mineclip}` respectively
  - VPT: [Model file](https://openaipublic.blob.core.windows.net/minecraft-rl/models/foundation-model-1x.model) and [Weight file](https://openaipublic.blob.core.windows.net/minecraft-rl/models/foundation-model-1x.weights)
  - MineCLIP: [`attn.pth`](https://drive.google.com/file/d/1uaZM1ZLBz2dZWcn85rZmjP7LV6Sg5PZW/view)
6. **TODO Submodule?** Download OpenAI VPT Repo and rename to `openai_vpt`

### Training

Run `python train.py --sample-size N` where N is the number of videos you want to train on.

This will generate 4 files in `./weights/ts_bc/` that store
- the latents from VPT in `latents_vpt.npy`,
- the latents from MineCLIP in `latents_mineclip.npy`,
- the actions in `actions.npy`, and
- the start indices of each used episode and its name in `episode_starts.npy`

### Inference

**TODO**
Run the agent: `python agent_vpt.py`.