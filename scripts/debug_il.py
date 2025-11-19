#!/usr/bin/env python3
"""Debug script to check IL training pipeline."""

import pickle
import numpy as np
import torch
import blosc

# Load demos
with open('demos/goto_redball_grey_10k.pkl', 'rb') as f:
    demos = pickle.load(f)

print(f"Loaded {len(demos)} demos")

# Check first demo structure
demo = demos[0]
print(f"\nDemo structure:")
print(f"  Mission: {demo[0]}")
print(f"  Images: {type(demo[1])}")
print(f"  Directions: {demo[2]}")
print(f"  Actions: {demo[3]}")

# Unpack images
images = blosc.unpack_array(demo[1])
print(f"\nImage shape: {images.shape}")
print(f"Image dtype: {images.dtype}")
print(f"Image min/max: {images.min()}, {images.max()}")

# Check action distribution
all_actions = []
for demo in demos[:1000]:  # First 1000 demos
    all_actions.extend(demo[3])

all_actions = np.array(all_actions)
action_names = ['left', 'right', 'forward', 'pickup', 'drop', 'toggle', 'done']
print(f"\nAction distribution (first 1000 demos):")
for i in range(7):
    count = (all_actions == i).sum()
    pct = 100 * count / len(all_actions)
    print(f"  {i} ({action_names[i]}): {count} ({pct:.1f}%)")

# Check if model can at least do a forward pass
print("\n--- Testing model forward pass ---")
from toddler_ai import utils
from toddler_ai.utils.demos import transform_demos

# Transform a small batch
batch = transform_demos(demos[:10])
print(f"Transformed batch: {len(batch)} episodes")
print(f"First episode length: {len(batch[0])}")
print(f"First step: obs keys = {batch[0][0][0].keys()}, action = {batch[0][0][1]}")

# Check observation format
obs = batch[0][0][0]
print(f"\nObservation:")
print(f"  image shape: {obs['image'].shape}")
print(f"  direction: {obs['direction']}")
print(f"  mission: {obs['mission']}")

# Load model and test
print("\n--- Testing model inference ---")
import os
import gymnasium as gym
if os.path.exists('models/p1_GoToRedBallGrey_vit'):
    env = gym.make('BabyAI-GoToRedBallGrey-v0')
    agent = utils.load_agent(
        env,
        model_name='p1_GoToRedBallGrey_vit',
        argmax=True
    )

    # Test on a few observations
    obss = [batch[0][i][0] for i in range(min(5, len(batch[0])))]
    actions_true = [batch[0][i][1] for i in range(min(5, len(batch[0])))]

    # Get model predictions
    device = utils.get_device()
    preprocessed = agent.obss_preprocessor(obss, device=device)

    with torch.no_grad():
        results = agent.model(preprocessed, agent.memory)
        probs = results['dist'].probs
        preds = probs.argmax(dim=1)

    print(f"\nModel predictions vs true actions:")
    for i in range(len(obss)):
        pred = preds[i].item()
        true = actions_true[i]
        prob_dist = probs[i].cpu().numpy()
        print(f"  Step {i}: pred={action_names[pred]}, true={action_names[true]}, probs={prob_dist.round(2)}")
else:
    print("No model found at models/p1_GoToRedBallGrey_vit")
