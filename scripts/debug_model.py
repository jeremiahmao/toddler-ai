#!/usr/bin/env python3
"""Debug script to trace values through the unified_vit model."""

import pickle
import numpy as np
import torch
import blosc
import gymnasium as gym

from toddler_ai import utils
from toddler_ai.utils.demos import transform_demos
from toddler_ai.utils.format import MiniLMPreprocessor
from toddler_ai.models.unified_vit_model import UnifiedViTACModel

# Load demos
print("Loading demos...")
with open('demos/goto_redball_grey_10k.pkl', 'rb') as f:
    demos = pickle.load(f)

print(f"Loaded {len(demos)} demos")

# Transform demos
batch = transform_demos(demos[:10])
print(f"Transformed {len(batch)} episodes")

# Create environment and preprocessor
env = gym.make('BabyAI-GoToRedBallGrey-v0')
obs_space = env.observation_space

# Create preprocessor (this is what format.py does)
preprocessor = MiniLMPreprocessor(
    model_name='test',
    obs_space=obs_space,
    model_name_minilm='prajjwal1/bert-tiny',
    freeze_encoder=False
)

print(f"\nPreprocessor obs_space: {preprocessor.obs_space}")

# Get a few observations
obss = [batch[0][i][0] for i in range(min(5, len(batch[0])))]
actions_true = [batch[0][i][1] for i in range(min(5, len(batch[0])))]

print(f"\nFirst observation:")
print(f"  image shape: {obss[0]['image'].shape}")
print(f"  image dtype: {obss[0]['image'].dtype}")
print(f"  image min/max: {obss[0]['image'].min()}, {obss[0]['image'].max()}")
print(f"  direction: {obss[0]['direction']}")
print(f"  mission: {obss[0]['mission']}")

# Preprocess observations
device = torch.device('cpu')
preprocessed = preprocessor(obss, device=device)

print(f"\n--- Preprocessed observations ---")
print(f"  image shape: {preprocessed.image.shape}")
print(f"  image dtype: {preprocessed.image.dtype}")
print(f"  image min/max: {preprocessed.image.min().item():.4f}, {preprocessed.image.max().item():.4f}")
print(f"  minilm_emb shape: {preprocessed.minilm_emb.shape}")
print(f"  minilm_emb dtype: {preprocessed.minilm_emb.dtype}")
print(f"  minilm_emb min/max: {preprocessed.minilm_emb.min().item():.4f}, {preprocessed.minilm_emb.max().item():.4f}")
print(f"  minilm_emb mean/std: {preprocessed.minilm_emb.mean().item():.4f}, {preprocessed.minilm_emb.std().item():.4f}")

# Create model
print("\n--- Creating model ---")
model = UnifiedViTACModel(
    obs_space=preprocessor.obs_space,
    action_space=env.action_space,
    embed_dim=256,
    use_memory=True,
    history_length=30
)

print(f"Model created")
print(f"  num_patches: {model.num_patches}")
print(f"  embed_dim: {model.embed_dim}")
print(f"  num_actions: {model.num_actions}")

# Create memory
batch_size = len(obss)
memory = torch.zeros(batch_size, model.history_length, dtype=torch.long)

# Forward pass with detailed logging
print("\n--- Forward pass ---")

# 1. Goal embedding
minilm_emb = preprocessed.minilm_emb
print(f"\n1. MiniLM embedding:")
print(f"   shape: {minilm_emb.shape}")
print(f"   min/max: {minilm_emb.min().item():.4f}, {minilm_emb.max().item():.4f}")
print(f"   mean/std: {minilm_emb.mean().item():.4f}, {minilm_emb.std().item():.4f}")

goal_embedding = model.minilm_projection(minilm_emb)
print(f"\n   After projection (128 -> 256):")
print(f"   shape: {goal_embedding.shape}")
print(f"   min/max: {goal_embedding.min().item():.4f}, {goal_embedding.max().item():.4f}")
print(f"   mean/std: {goal_embedding.mean().item():.4f}, {goal_embedding.std().item():.4f}")

goal_token = goal_embedding.unsqueeze(1)

# 2. Vision patches
image = preprocessed.image.permute(0, 3, 1, 2).float()
print(f"\n2. Image tensor:")
print(f"   shape: {image.shape}")
print(f"   min/max: {image.min().item():.4f}, {image.max().item():.4f}")
print(f"   mean/std: {image.mean().item():.4f}, {image.std().item():.4f}")

vision_patches = model.patch_embed(image)
print(f"\n   After patch embedding:")
print(f"   shape: {vision_patches.shape}")
print(f"   min/max: {vision_patches.min().item():.4f}, {vision_patches.max().item():.4f}")
print(f"   mean/std: {vision_patches.mean().item():.4f}, {vision_patches.std().item():.4f}")

# 3. Action history
action_history = model.action_embeddings(memory.long())
action_history = action_history + model.temporal_pos_embeddings
print(f"\n3. Action history:")
print(f"   shape: {action_history.shape}")
print(f"   min/max: {action_history.min().item():.4f}, {action_history.max().item():.4f}")
print(f"   mean/std: {action_history.mean().item():.4f}, {action_history.std().item():.4f}")

# 4. Context buffer
context = torch.cat([goal_token, action_history, vision_patches], dim=1)
print(f"\n4. Context buffer (concatenated):")
print(f"   shape: {context.shape}")
print(f"   min/max: {context.min().item():.4f}, {context.max().item():.4f}")
print(f"   mean/std: {context.mean().item():.4f}, {context.std().item():.4f}")

# 5. Self-attention
for i, attn_layer in enumerate(model.context_attention):
    context = attn_layer(context)
    print(f"\n5.{i+1}. After attention layer {i+1}:")
    print(f"   shape: {context.shape}")
    print(f"   min/max: {context.min().item():.4f}, {context.max().item():.4f}")
    print(f"   mean/std: {context.mean().item():.4f}, {context.std().item():.4f}")

# 6. Pooling
state_concept = context.mean(dim=1)
print(f"\n6. After mean pooling:")
print(f"   shape: {state_concept.shape}")
print(f"   min/max: {state_concept.min().item():.4f}, {state_concept.max().item():.4f}")
print(f"   mean/std: {state_concept.mean().item():.4f}, {state_concept.std().item():.4f}")

state_concept = model.pool(state_concept)
print(f"\n   After pool layer (with Tanh):")
print(f"   shape: {state_concept.shape}")
print(f"   min/max: {state_concept.min().item():.4f}, {state_concept.max().item():.4f}")
print(f"   mean/std: {state_concept.mean().item():.4f}, {state_concept.std().item():.4f}")

# 7. Actor output
action_logits = model.actor(state_concept)
print(f"\n7. Action logits:")
print(f"   shape: {action_logits.shape}")
print(f"   min/max: {action_logits.min().item():.4f}, {action_logits.max().item():.4f}")
print(f"   mean/std: {action_logits.mean().item():.4f}, {action_logits.std().item():.4f}")
print(f"   values: {action_logits[0].detach().numpy().round(3)}")

# Check distribution
from torch.distributions.categorical import Categorical
import torch.nn.functional as F

dist = Categorical(logits=action_logits)
probs = dist.probs
print(f"\n   Probabilities (from Categorical):")
print(f"   shape: {probs.shape}")
print(f"   values: {probs[0].detach().numpy().round(3)}")
print(f"   sum: {probs[0].sum().item():.4f}")

# Compare with manual softmax
manual_probs = F.softmax(action_logits, dim=1)
print(f"\n   Manual softmax probs: {manual_probs[0].detach().numpy().round(3)}")

# Check what the OLD code was doing (double log-softmax)
print(f"\n--- What the BUG was doing ---")
log_softmax_logits = F.log_softmax(action_logits, dim=1)
print(f"   log_softmax(logits): {log_softmax_logits[0].detach().numpy().round(3)}")

bad_dist = Categorical(logits=log_softmax_logits)
bad_probs = bad_dist.probs
print(f"   Categorical(logits=log_softmax(logits)).probs: {bad_probs[0].detach().numpy().round(3)}")
print(f"   This is softmax(log_softmax(x)) - nearly uniform!")

# Compare entropy
print(f"\n   Correct entropy: {dist.entropy()[0].item():.4f}")
print(f"   Buggy entropy: {bad_dist.entropy()[0].item():.4f}")

# Full forward pass
print("\n--- Full forward pass ---")
with torch.no_grad():
    result = model(preprocessed, memory)

print(f"Distribution probs: {result['dist'].probs[0].detach().numpy().round(3)}")
print(f"Value: {result['value'][0].item():.4f}")
print(f"Entropy: {result['dist'].entropy()[0].item():.4f}")

# Check predictions
print(f"\nTrue actions: {actions_true}")
preds = result['dist'].probs.argmax(dim=1)
print(f"Predicted actions: {preds.tolist()}")

action_names = ['left', 'right', 'forward', 'pickup', 'drop', 'toggle', 'done']
print(f"\nAction predictions:")
for i in range(len(obss)):
    pred = preds[i].item()
    true = actions_true[i]
    probs_i = result['dist'].probs[i].detach().numpy()
    print(f"  Step {i}: pred={action_names[pred]}, true={action_names[true]}")
    print(f"           probs={probs_i.round(3)}")
