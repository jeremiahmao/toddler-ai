#!/usr/bin/env python3
"""Test that PPO now properly trains bert-tiny with the complete fix.

This verifies:
1. PPO runs without graph reuse errors
2. bert-tiny parameters are included in optimizer
3. bert-tiny receives gradients and updates during training

Run with: uv run python scripts/test_ppo_bert_training.py
"""

import torch
import gymnasium as gym
from toddler_ai.utils.format import MiniLMPreprocessor
from toddler_ai.models.unified_vit_model import UnifiedViTACModel
from toddler_ai.algorithms.ppo import PPOAlgo
from toddler_ai.utils.penv import ParallelEnv


def test_ppo_bert_training():
    """Test that PPO properly trains bert-tiny after our fixes."""
    print("=" * 60)
    print("Testing Complete PPO Fix: bert-tiny Training")
    print("=" * 60)

    # Setup environment
    envs = []
    for _ in range(2):  # Small number for testing
        envs.append(gym.make('BabyAI-GoToRedBallGrey-v0'))

    # Create preprocessor with trainable bert-tiny
    preprocessor = MiniLMPreprocessor(
        model_name='test',
        obs_space=envs[0].observation_space,
        model_name_minilm='prajjwal1/bert-tiny',
        freeze_encoder=False  # TRAINABLE
    )

    # Create model (use CPU for testing)
    model = UnifiedViTACModel(
        obs_space=preprocessor.obs_space,
        action_space=envs[0].action_space,
        embed_dim=256,
        use_memory=True,
        history_length=10
    )
    # No need to move to device - PPO will handle it

    # Create PPO algorithm
    algo = PPOAlgo(
        envs,
        model,
        num_frames_per_proc=128,  # Small for testing
        discount=0.99,
        lr=1e-4,
        gae_lambda=0.95,
        entropy_coef=0.01,
        value_loss_coef=0.5,
        max_grad_norm=0.5,
        recurrence=4,
        adam_eps=1e-8,
        clip_eps=0.2,
        epochs=4,  # The critical part - multiple epochs
        batch_size=32,
        preprocess_obss=preprocessor,
        reshape_reward=None
    )

    # Track bert-tiny parameters before training
    bert_params = list(preprocessor.minilm_encoder.parameters())
    if not bert_params:
        print("✗ ERROR: No bert-tiny parameters found!")
        return False

    bert_param = bert_params[0]
    initial_bert_value = bert_param.data.clone()

    print(f"\nSetup complete:")
    print(f"  Device: {algo.device}")
    print(f"  Envs: {algo.num_procs} parallel")
    print(f"  PPO epochs: {algo.epochs}")
    print(f"  Frames per proc: {algo.num_frames_per_proc}")
    print(f"  bert-tiny trainable: {not preprocessor.freeze_encoder}")
    print(f"  bert-tiny params: {sum(p.numel() for p in bert_params):,}")
    print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Check optimizer setup
    print(f"\nOptimizer check:")
    optimizer_params = sum(sum(p.numel() for p in group['params'])
                          for group in algo.optimizer.param_groups)
    expected_params = (sum(p.numel() for p in model.parameters()) +
                      sum(p.numel() for p in bert_params))
    print(f"  Total params in optimizer: {optimizer_params:,}")
    print(f"  Expected (model + bert): {expected_params:,}")

    if optimizer_params != expected_params:
        print(f"  ✗ MISMATCH: Optimizer missing parameters!")
        return False
    print(f"  ✓ Optimizer includes all parameters")

    # Run training for a few updates
    print(f"\nRunning PPO updates...")
    num_updates = 3

    for update in range(num_updates):
        # Collect experiences
        exps, logs = algo.collect_experiences()

        # Update parameters (this is where the magic happens)
        update_logs = algo.update_parameters()

        print(f"\nUpdate {update + 1}:")
        if logs.get('return_per_episode'):
            print(f"  Return: {logs['return_per_episode']}")
        print(f"  Loss: {update_logs['loss']:.4f}")
        print(f"  Policy loss: {update_logs['policy_loss']:.4f}")
        print(f"  Value loss: {update_logs['value_loss']:.4f}")
        print(f"  Entropy: {update_logs['entropy']:.4f}")
        print(f"  Grad norm: {update_logs['grad_norm']:.4f}")

        # Check if bert-tiny is changing
        bert_change = (bert_param.data - initial_bert_value).abs().max().item()
        print(f"  bert-tiny max change: {bert_change:.6f}")

    # Final check
    total_bert_change = (bert_param.data - initial_bert_value).abs().sum().item()
    print(f"\n" + "=" * 60)
    print(f"RESULTS:")
    print(f"  Total bert-tiny parameter change: {total_bert_change:.6f}")

    if total_bert_change > 1e-6:
        print(f"  ✓ SUCCESS: bert-tiny is being trained during PPO!")
        print(f"  ✓ No graph reuse errors!")
        print(f"  ✓ Complete fix verified!")
        return True
    else:
        print(f"  ✗ FAIL: bert-tiny parameters not changing")
        return False


def main():
    try:
        success = test_ppo_bert_training()
    except Exception as e:
        print(f"\n✗ ERROR during test: {e}")
        import traceback
        traceback.print_exc()
        success = False

    print("\n" + "=" * 60)
    print(f"Final Result: {'PASS - PPO properly trains bert-tiny!' if success else 'FAIL'}")
    print("=" * 60)

    if success:
        print("\nThe complete fix includes:")
        print("1. Storing raw observations separately from DictList")
        print("2. Re-preprocessing observations for each PPO epoch/batch")
        print("3. Including bert-tiny in PPO optimizer with differential LR")
        print("\nPPO now trains ALL components, matching IL training!")


if __name__ == '__main__':
    main()