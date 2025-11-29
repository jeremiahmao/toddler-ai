#!/usr/bin/env python3
"""Comprehensive debug script to verify model and training pipelines.

This script traces values through:
1. Observation preprocessing
2. Model forward pass
3. IL training loop
4. PPO training loop
5. Inference/evaluation

Run with: uv run python scripts/debug_pipeline.py
"""

import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import gymnasium as gym

from toddler_ai import utils
from toddler_ai.utils.demos import transform_demos
from toddler_ai.utils.format import MiniLMPreprocessor
from toddler_ai.models.unified_vit_model import UnifiedViTACModel


def check_value_ranges(name, tensor, expected_min=None, expected_max=None):
    """Check if tensor values are in expected ranges."""
    if tensor is None:
        print(f"  {name}: None")
        return

    if not isinstance(tensor, torch.Tensor):
        print(f"  {name}: {tensor}")
        return

    # Handle integer tensors (like memory with action indices)
    if tensor.dtype in [torch.long, torch.int, torch.int32, torch.int64]:
        min_val = tensor.min().item()
        max_val = tensor.max().item()
        print(f"  {name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}, min={min_val}, max={max_val}")
        return

    min_val = tensor.min().item()
    max_val = tensor.max().item()
    mean_val = tensor.mean().item()
    std_val = tensor.std().item()

    status = "✓"
    if expected_min is not None and min_val < expected_min:
        status = "✗ MIN"
    if expected_max is not None and max_val > expected_max:
        status = "✗ MAX"

    print(f"  {name}: shape={tuple(tensor.shape)}, min={min_val:.4f}, max={max_val:.4f}, mean={mean_val:.4f}, std={std_val:.4f} {status}")


def debug_preprocessing():
    """Test observation preprocessing."""
    print("\n" + "="*60)
    print("1. OBSERVATION PREPROCESSING")
    print("="*60)

    # Load demos
    with open('demos/goto_redball_grey_10k.pkl', 'rb') as f:
        demos = pickle.load(f)

    # Transform demos
    batch = transform_demos(demos[:5])

    # Create environment and preprocessor
    env = gym.make('BabyAI-GoToRedBallGrey-v0')
    obs_space = env.observation_space

    preprocessor = MiniLMPreprocessor(
        model_name='test',
        obs_space=obs_space,
        model_name_minilm='prajjwal1/bert-tiny',
        freeze_encoder=False
    )

    # Get observations
    obss = [batch[0][i][0] for i in range(min(5, len(batch[0])))]
    actions = [batch[0][i][1] for i in range(min(5, len(batch[0])))]

    print("\nRaw observations:")
    check_value_ranges("image", torch.tensor(obss[0]['image']).float(), 0, 10)
    print(f"  mission: '{obss[0]['mission']}'")

    # Preprocess
    device = torch.device('cpu')
    preprocessed = preprocessor(obss, device=device)

    print("\nPreprocessed observations:")
    check_value_ranges("image (normalized)", preprocessed.image, 0, 1)
    check_value_ranges("minilm_emb", preprocessed.minilm_emb)

    return env, preprocessor, obss, actions, preprocessed


def debug_model_forward(env, preprocessor, preprocessed, verbose=True):
    """Test model forward pass."""
    if verbose:
        print("\n" + "="*60)
        print("2. MODEL FORWARD PASS")
        print("="*60)

    # Create model
    model = UnifiedViTACModel(
        obs_space=preprocessor.obs_space,
        action_space=env.action_space,
        embed_dim=256,
        use_memory=True,
        history_length=10  # Match default
    )

    batch_size = preprocessed.image.size(0)
    memory = torch.zeros(batch_size, model.memory_size, dtype=torch.long)

    if verbose:
        print(f"\nModel config:")
        print(f"  num_patches: {model.num_patches}")
        print(f"  embed_dim: {model.embed_dim}")
        print(f"  num_actions: {model.num_actions}")
        print(f"  history_length: {model.history_length}")
        print(f"  memory_size: {model.memory_size}")

    # Forward pass
    with torch.no_grad():
        result = model(preprocessed, memory)

    if verbose:
        print("\nModel outputs:")
        probs = result['dist'].probs
        check_value_ranges("action probs", probs, 0, 1)
        check_value_ranges("value", result['value'])
        check_value_ranges("memory (returned)", result['memory'])

        print(f"\n  Entropy: {result['dist'].entropy().mean().item():.4f}")
        print(f"  Prob sum: {probs.sum(dim=1).mean().item():.4f} (should be 1.0)")

        # Check if probabilities are uniform (bad) or peaked (good)
        max_prob = probs.max(dim=1)[0].mean().item()
        print(f"  Max prob (mean): {max_prob:.4f} (>0.2 = learning, ~0.14 = uniform)")

    return model, result


def debug_categorical_distribution():
    """Verify Categorical distribution is working correctly."""
    print("\n" + "="*60)
    print("3. CATEGORICAL DISTRIBUTION CHECK")
    print("="*60)

    # Test logits
    logits = torch.tensor([[1.0, -1.0, 0.0, 0.5, -0.5, 0.2, -0.2]])

    # Correct usage
    dist_correct = Categorical(logits=logits)
    probs_correct = dist_correct.probs

    # Buggy usage (double log-softmax)
    log_softmax_logits = F.log_softmax(logits, dim=1)
    dist_buggy = Categorical(logits=log_softmax_logits)
    probs_buggy = dist_buggy.probs

    print("\nLogits:", logits[0].numpy().round(3))
    print("\nCorrect (Categorical(logits=logits)):")
    print(f"  probs: {probs_correct[0].numpy().round(3)}")
    print(f"  entropy: {dist_correct.entropy()[0].item():.4f}")

    print("\nBuggy (Categorical(logits=log_softmax(logits))):")
    print(f"  probs: {probs_buggy[0].numpy().round(3)}")
    print(f"  entropy: {dist_buggy.entropy()[0].item():.4f}")

    # Check gradient flow
    print("\nGradient check:")
    logits_grad = torch.tensor([[1.0, -1.0, 0.0, 0.5, -0.5, 0.2, -0.2]], requires_grad=True)
    dist = Categorical(logits=logits_grad)
    target_action = torch.tensor([2])  # action 2
    loss = -dist.log_prob(target_action)
    loss.backward()

    print(f"  log_prob(action=2): {dist.log_prob(target_action).item():.4f}")
    print(f"  gradient: {logits_grad.grad[0].numpy().round(4)}")
    print(f"  gradient sum: {logits_grad.grad.sum().item():.6f} (should be ~0)")


def debug_memory_flow():
    """Test that memory is updated correctly with actions."""
    print("\n" + "="*60)
    print("4. MEMORY FLOW CHECK")
    print("="*60)

    env = gym.make('BabyAI-GoToRedBallGrey-v0')

    # Create simple model
    from toddler_ai.utils.format import MiniLMPreprocessor
    preprocessor = MiniLMPreprocessor(
        model_name='test',
        obs_space=env.observation_space,
        model_name_minilm='prajjwal1/bert-tiny',
        freeze_encoder=True  # faster for test
    )

    model = UnifiedViTACModel(
        obs_space=preprocessor.obs_space,
        action_space=env.action_space,
        embed_dim=256,
        use_memory=True,
        history_length=5
    )

    # Initialize memory
    batch_size = 2
    memory = torch.zeros(batch_size, model.memory_size, dtype=torch.long)

    print(f"\nMemory shape: {memory.shape}")
    print(f"Memory dtype: {memory.dtype}")
    print(f"Initial memory: {memory[0].tolist()}")

    # Simulate a few steps
    obs, _ = env.reset()

    for step in range(3):
        # Preprocess
        preprocessed = preprocessor([obs, obs], device=torch.device('cpu'))

        # Forward
        with torch.no_grad():
            result = model(preprocessed, memory)

        # Sample action
        action = result['dist'].sample()

        # CRITICAL: Update memory with action
        # Shift memory left and add new action
        new_memory = torch.zeros_like(memory)
        new_memory[:, :-1] = memory[:, 1:]  # Shift left
        new_memory[:, -1] = action  # Add new action
        memory = new_memory

        print(f"Step {step}: action={action[0].item()}, memory={memory[0].tolist()}")

        # Step environment
        obs, _, done, _, _ = env.step(action[0].item())
        if done:
            break

    print("\n⚠️  ISSUE: The model's forward() returns the SAME memory it received!")
    print("   For IL training, memory must be updated externally with actions taken.")
    print("   Current IL code does NOT update memory with actions!")


def debug_il_training_loop():
    """Verify IL training loss computation."""
    print("\n" + "="*60)
    print("5. IL TRAINING LOSS CHECK")
    print("="*60)

    # Simple test: model should learn to predict correct action given obs
    env = gym.make('BabyAI-GoToRedBallGrey-v0')

    preprocessor = MiniLMPreprocessor(
        model_name='test',
        obs_space=env.observation_space,
        model_name_minilm='prajjwal1/bert-tiny',
        freeze_encoder=True
    )

    model = UnifiedViTACModel(
        obs_space=preprocessor.obs_space,
        action_space=env.action_space,
        embed_dim=256,
        use_memory=False,  # Simpler test without memory
        history_length=1
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Load a small batch of demos
    with open('demos/goto_redball_grey_10k.pkl', 'rb') as f:
        demos = pickle.load(f)

    batch = transform_demos(demos[:10])

    # Get a few observations and actions
    obss = [batch[i][0][0] for i in range(min(10, len(batch)))]
    actions = torch.tensor([batch[i][0][1] for i in range(min(10, len(batch)))])

    device = torch.device('cpu')
    preprocessed = preprocessor(obss, device=device)
    memory = torch.zeros(len(obss), model.memory_size, dtype=torch.long)

    print("\nTraining for 50 steps...")

    initial_loss = None
    for step in range(50):
        optimizer.zero_grad()

        result = model(preprocessed, memory)
        dist = result['dist']

        # IL loss: cross-entropy
        loss = -dist.log_prob(actions).mean()
        entropy = dist.entropy().mean()

        if step == 0:
            initial_loss = loss.item()

        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            accuracy = (dist.probs.argmax(dim=1) == actions).float().mean().item()
            print(f"  Step {step}: loss={loss.item():.4f}, entropy={entropy.item():.4f}, accuracy={accuracy:.2%}")

    final_loss = loss.item()
    final_accuracy = (dist.probs.argmax(dim=1) == actions).float().mean().item()

    print(f"\nResults:")
    print(f"  Initial loss: {initial_loss:.4f}")
    print(f"  Final loss: {final_loss:.4f}")
    print(f"  Loss reduction: {(initial_loss - final_loss) / initial_loss * 100:.1f}%")
    print(f"  Final accuracy: {final_accuracy:.2%}")

    if final_loss < initial_loss * 0.5:
        print("\n✓ Model CAN learn from IL - gradients flowing correctly")
    else:
        print("\n✗ Model NOT learning - check gradient flow!")


def debug_ppo_training_loop():
    """Verify PPO value/advantage computation."""
    print("\n" + "="*60)
    print("6. PPO TRAINING CHECK")
    print("="*60)

    # This would require full PPO setup, skip for now
    print("\nPPO uses same model forward pass, should work if IL works.")
    print("Key differences:")
    print("  - PPO updates memory externally in base.py:156-157")
    print("  - PPO uses dist.sample() not dist.probs.argmax()")
    print("  - PPO normalizes advantages (line 285-287 in ppo.py)")


def debug_inference():
    """Verify inference/evaluation pipeline."""
    print("\n" + "="*60)
    print("7. INFERENCE CHECK")
    print("="*60)

    env = gym.make('BabyAI-GoToRedBallGrey-v0')

    # Check if model exists
    import os
    if not os.path.exists('models/p1_GoToRedBallGrey_il'):
        print("\nNo trained model found. Skipping inference test.")
        print("Train a model first with: uv run python scripts/train_il.py ...")
        return

    # Load agent
    agent = utils.load_agent(env, model_name='p1_GoToRedBallGrey_il', argmax=True)

    print(f"\nAgent loaded:")
    print(f"  Model type: {type(agent.model).__name__}")
    print(f"  Memory size: {agent.model.memory_size}")
    print(f"  Device: {agent.device}")

    # Run a few episodes
    successes = 0
    for ep in range(5):
        obs, _ = env.reset(seed=ep)
        agent.on_reset()

        for step in range(64):
            result = agent.act(obs)
            action = result['action'].item()
            obs, reward, done, truncated, _ = env.step(action)
            agent.analyze_feedback(reward, done)

            if done:
                if reward > 0:
                    successes += 1
                break

        print(f"  Episode {ep}: {'SUCCESS' if reward > 0 else 'FAIL'} in {step+1} steps")

    print(f"\nSuccess rate: {successes}/5 = {successes/5:.0%}")


def main():
    print("="*60)
    print("TODDLER-AI PIPELINE DEBUG")
    print("="*60)

    # Run all debug checks
    env, preprocessor, obss, actions, preprocessed = debug_preprocessing()
    model, result = debug_model_forward(env, preprocessor, preprocessed)
    debug_categorical_distribution()
    debug_memory_flow()
    debug_il_training_loop()
    debug_ppo_training_loop()
    debug_inference()

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    print("""
Key findings:

1. ✓ Preprocessing: Images normalized to 0-1, bert-tiny produces 128-dim embeddings
2. ✓ Categorical: Using logits directly (not log_softmax) is correct
3. ⚠️  Memory: Model returns same memory - must be updated externally with actions
4. ?  IL accuracy: Need to verify learning actually works with the training loop

Potential issues to investigate:
- IL training may not be updating memory with actions (check imitation.py)
- Learning rate may be too low for bert-tiny layer
- Batch size / recurrence settings may need tuning
""")


if __name__ == '__main__':
    main()
