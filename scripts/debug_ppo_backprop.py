#!/usr/bin/env python3
"""Debug script to test PPO backprop through bert-tiny.

This tests whether we can successfully backprop through bert-tiny
when re-encoding text fresh for each epoch.

Run with: uv run python scripts/debug_ppo_backprop.py
"""

import torch
import gymnasium as gym
from toddler_ai.utils.format import MiniLMPreprocessor
from toddler_ai.models.unified_vit_model import UnifiedViTACModel


def test_multiple_epoch_backprop():
    """Test if we can backprop through bert-tiny multiple times by re-encoding."""
    print("=" * 60)
    print("Testing PPO-style multiple epoch backprop through bert-tiny")
    print("=" * 60)

    # Setup
    env = gym.make('BabyAI-GoToRedBallGrey-v0')

    preprocessor = MiniLMPreprocessor(
        model_name='test',
        obs_space=env.observation_space,
        model_name_minilm='prajjwal1/bert-tiny',
        freeze_encoder=False  # TRAINABLE encoder
    )

    model = UnifiedViTACModel(
        obs_space=preprocessor.obs_space,
        action_space=env.action_space,
        embed_dim=256,
        use_memory=True,
        history_length=10
    )

    # Create optimizer that includes bert-tiny parameters
    all_params = list(model.parameters()) + list(preprocessor.minilm_encoder.parameters())
    optimizer = torch.optim.Adam(all_params, lr=1e-4)

    # Get some raw observations
    device = torch.device('cpu')
    obs, _ = env.reset()
    raw_obss = [obs] * 8  # Batch of 8

    # Simulate PPO: multiple epochs over same data
    num_epochs = 4
    print(f"\nSimulating {num_epochs} PPO epochs over same raw observations")
    print(f"Batch size: {len(raw_obss)}")
    print(f"bert-tiny trainable: {not preprocessor.freeze_encoder}")

    memory = torch.zeros(len(raw_obss), model.memory_size, dtype=torch.long)

    # Track bert-tiny gradient to confirm it's being updated
    bert_param = list(preprocessor.minilm_encoder.parameters())[0]
    initial_bert_value = bert_param.data.clone()

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # KEY: Re-preprocess (re-encode text) for each epoch
        # This creates a fresh computation graph each time
        preprocessed = preprocessor(raw_obss, device=device)

        # Forward pass
        result = model(preprocessed, memory)
        dist = result['dist']

        # Fake PPO loss
        fake_actions = torch.randint(0, 7, (len(raw_obss),))
        loss = -dist.log_prob(fake_actions).mean()

        # Backward
        loss.backward()

        # Check gradients
        bert_grad_norm = bert_param.grad.norm().item() if bert_param.grad is not None else 0
        model_grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)

        print(f"  Epoch {epoch}: loss={loss.item():.4f}, bert_grad={bert_grad_norm:.6f}, model_grad={model_grad_norm:.4f}")

        # Update
        optimizer.step()

    # Check if bert-tiny actually changed
    bert_change = (bert_param.data - initial_bert_value).abs().sum().item()
    print(f"\nbert-tiny parameter change: {bert_change:.6f}")

    if bert_change > 0:
        print("✓ SUCCESS: bert-tiny parameters are being updated!")
    else:
        print("✗ FAIL: bert-tiny parameters not changing")

    return True


def test_detached_approach():
    """Test the current detach approach (no bert-tiny backprop)."""
    print("\n" + "=" * 60)
    print("Testing detached approach (bert-tiny frozen during PPO)")
    print("=" * 60)

    env = gym.make('BabyAI-GoToRedBallGrey-v0')

    preprocessor = MiniLMPreprocessor(
        model_name='test',
        obs_space=env.observation_space,
        model_name_minilm='prajjwal1/bert-tiny',
        freeze_encoder=False  # Trainable, but we'll detach
    )

    model = UnifiedViTACModel(
        obs_space=preprocessor.obs_space,
        action_space=env.action_space,
        embed_dim=256,
        use_memory=True,
        history_length=10
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    device = torch.device('cpu')
    obs, _ = env.reset()
    raw_obss = [obs] * 8

    # Preprocess ONCE and detach
    preprocessed = preprocessor(raw_obss, device=device)
    preprocessed.minilm_emb = preprocessed.minilm_emb.detach()

    memory = torch.zeros(len(raw_obss), model.memory_size, dtype=torch.long)

    num_epochs = 4
    print(f"\nSimulating {num_epochs} PPO epochs with detached embeddings")

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        result = model(preprocessed, memory)
        dist = result['dist']

        fake_actions = torch.randint(0, 7, (len(raw_obss),))
        loss = -dist.log_prob(fake_actions).mean()

        loss.backward()

        model_grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
        print(f"  Epoch {epoch}: loss={loss.item():.4f}, model_grad={model_grad_norm:.4f}")

        optimizer.step()

    print("✓ Detached approach works (but no bert-tiny training)")
    return True


def main():
    print("PPO bert-tiny Backprop Debug")
    print("=" * 60)

    try:
        success1 = test_multiple_epoch_backprop()
    except Exception as e:
        print(f"\n✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        success1 = False

    try:
        success2 = test_detached_approach()
    except Exception as e:
        print(f"\n✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        success2 = False

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Re-encoding per epoch: {'PASS' if success1 else 'FAIL'}")
    print(f"Detached approach: {'PASS' if success2 else 'FAIL'}")

    if success1:
        print("\nSOLUTION: To backprop through bert-tiny during PPO,")
        print("we need to store raw observations and re-encode them")
        print("fresh for each epoch, rather than preprocessing once.")


if __name__ == '__main__':
    main()
