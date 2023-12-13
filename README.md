# TS-BC

**Targeted Search-Based Behavioral-Cloning** for learning to play Minecraft.

**Search-Based Behavioral-Cloning** (S-BC) searches the latent space of a neural model (here VPT) for similar situations in the dataset.
After finding a suitable starting point it copies the actions that were done in the demonstrator dataset.

**Targeted S-BC** (TS-BC) takes a text goal and searches the dataset for similar outcomes in another model (here MineCLIP).
The search results from both models are then filtered to get trajectories where the current agent situation can lead to the goal.
The actions from those trajectories are then copied.

Videos of test runs with different prompts copied from [STEVE-1](https://sites.google.com/view/steve-1) can be found here: [Drive](https://drive.google.com/drive/folders/1kM6IpEP3bAnmKYh3X5_NXApNAsf6za_2?usp=drive_link)

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
5. Download weights and put them into `./weights/{vpt or mineclip or cvae}` respectively
  - VPT: [Model file](https://openaipublic.blob.core.windows.net/minecraft-rl/models/foundation-model-1x.model) and [Weight file](https://openaipublic.blob.core.windows.net/minecraft-rl/models/foundation-model-1x.weights)
  - MineCLIP: [`attn.pth`](https://drive.google.com/file/d/1uaZM1ZLBz2dZWcn85rZmjP7LV6Sg5PZW/view)
  - CVAE (from Steve-1): [`steve1_prior.pt`](https://drive.google.com/uc?id=1OdX5wiybK8jALVfP5_dEo0CWm9BQbDES)
6. After cloning this repository, clone the OpenAI VPT submodule using
```
git submodule init
git submodule update
```
7. ~~**TODO Umap** `pip install numba scikit-learn umap-learn umap-learn[plot]`~~ (later for analysing)

---

### Run the model

##### Training

Run `python train.py --sample-size N` where N is the number of videos you want to train on.

This will generate 4 files in `./weights/ts_bc/` that store
- the latents from VPT in `latents_vpt.npy`,
- the latents from MineCLIP in `latents_mineclip.npy`,
- the actions in `actions.npy`, and
- the start indices of each used episode and its name in `episode_starts.npy`

##### Inference

Run the agent: `python run_agent.py`.

**Note:** If you run the agent on a headless machine, use `xvfb-run python run_agent.py`

##### Analysis

Using: `python analyse.py` after you run the model, you can create a analysis video where at the top you can see the recorded agent video and at the bottom the current frames from the dataset video the agent is currently copying from. Also at the bottom you can see the computed difference score between the agent frames and dataset video frames.

---

### How the model works

The basic idea is to calculate the **similarity between the current agent state and all frames of all videos** from the dataset.
Searching these similarities you can find situations from the dataset that are similar to the current state which lets the agent **copy the next actions** done in the dataset.
Using the MineCLIP text encoder the agent **searches for dataset episodes where a given goal is reached** and combines them with the episodes that are similar to the agent to find a trajectory that would **bring the agent close to the given goal**.

##### # Latent point cloud

To compute the similarities, every frame from the dataset as well as at inference from the agent will be transformed into a latent representation by the MineCLIP video encoder.
These 512-dim points are saved in an array `latents_mineclip` of all frames from all episodes of shape `(N_episodes * Frames_per_episode, 512)`.
Another array `episode_starts` will keep track on where episodes end while an `actions` array saves the actions made between the frames.

##### # The search algorithm

During inference, when the gym environment asks for the next action, the agent is presented with the latent representation of its current state (`latent`) and the latent representation of every dataset frame. It now computes the distances from its latent to every dataset frame in `LatentSpaceMineCLIP::get_distances(latent)`. In standard, non-targeted S-BC the agent would choose the argmin of these distances to get the closest point and copy the actions done from this frame until a new search is done.

In targeted S-BC, the agent will also take the goal into account. For that it computes a so called `future_goal_distances` array when getting a new text goal in `TargetedSearchAgent::set_goal(text_goal)`. Using MineCLIP text encoder, a latent representation of the goal is created which is used to compute its distance to every frame latent as before. This `goal_distances` array now shows, how near every frame is to the specified goal. Every entry of this array will now be set to the minimum of the distances of its next N frames, thus creating the `future_goal_distances` array which states how far near the goal the agent could get in the next seconds if it follows a perticular trajectory of the dataset.

When searching for a trajectory in `TargetedSearchAgent::search(latent)`, instead of just using argmin on the agent state distance array, the `future_goal_distances` will first be added to it. Now the argmin selects a frame of an episode where the current state is similar and where it gets as close as possible to the goal in the next seconds.

##### # Penalty for selecting the same episode

In the search code there is another array added onto the distances before using argmin: `same_episode_penalty`.
This mitigates directly selecting the same episode multiple times by adding a penalty value (e.g. 10) onto the newly selected frame and its surroundings.
Every search, this penalty is decreased so that the episode can be selected again in the future, after others have been selected.

##### # Redoing the search

After a trajectory has been found, the agent would follow the dataset frames and copy their actions one by one until a new search is done.
There are 4 conditions seen in `TargetedSearchAgent::should_search_again()` that result in searching the dataset again:
1. **Initial:** Obviously when there hasn't been a search before.
2. **Timeout:** The parameter `max_follow_frames` controlls, how long a trajectory can be followed before the search has to be done again.
3. **Episode end:** When the episode from the dataset ends, a new search has to take place.
4. **Divergence:** If the difference from the current state to the frame of the episode the agent currently follow exceeds a certain threshold, the search has to be done again.

---

###### Detailed explanations

- Computing the `future_goal_distances`

Let's say this is your `goal_distances` array: `[11, 10, 9, 8, 7, 11, 10]`, meaning at the first 5 frames you gradually approach the goal but at frame 6 you get a bit more away from the goal but then getting closer again. If the window size to look into the future (`goal_rolling_window_size`) is set to `3`, the `future_goal_distances` array will look like this: `[9, 8, 7, 7, 7, 10, 10]`, meaning if you start following this trajectory on frame 1 you can get a distance of 9 near the goal in the next `3` frames. If you choose the 3rd frame, you could get as close as 7 to your goal.

- Searching after divergence

In order to stop rapidly changing the trajectory a counter `redo_search_counter` is increased by 1 every time, the difference exceeds a threshold. Only when this counter reaches a certain threshold itself (`redo_search_threshold`, e.g. 5), a new search should be done and the counter will be resetted.
Also, for the first few frames, after a new trajectory has been chosen, the counter won't be increased.