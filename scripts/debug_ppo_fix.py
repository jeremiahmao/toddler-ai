#!/usr/bin/env python3
"""Test that the PPO fix works with actual PPO algorithm.

Run with: uv run python scripts/debug_ppo_fix.py
"""

import torch
import gymnasium as gym
from toddler_ai.utils.format import MiniLMPreprocessor
from toddler_ai.models.unified_vit_model import UnifiedViTACModel
from toddler_ai.utils.penv import ParallelEnv


def test_ppo_with_fix():
    """Test PPO with re-encoding per epoch."""
    print("=" * 60)
    print("Testing PPO with re-encoding fix")
    print("=" * 60)

    # Use CPU for testing
    device = torch.device('cpu')

    # Create single environment
    env = gym.make('BabyAI-GoToRedBallGrey-v0')

    # Create preprocessor
    preprocessor = MiniLMPreprocessor(
        model_name='test',
        obs_space=env.observation_space,
        model_name_minilm='prajjwal1/bert-tiny',
        freeze_encoder=False  # TRAINABLE
    )

    # Create model
    model = UnifiedViTACModel(
        obs_space=preprocessor.obs_space,
        action_space=env.action_space,
        embed_dim=256,
        use_memory=True,
        history_length=10
    )

    # Get some raw observations
    obs, _ = env.reset()
    raw_obss = [obs] * 8

    # Simulate PPO epoch loop with re-encoding
    from toddler_ai.utils.dictlist import DictList
    exps = DictList()
    exps.raw_obs = raw_obss
    exps.memory = torch.zeros(8, model.memory_size, dtype=torch.long)
    exps.mask = torch.ones(8, 1)
    exps.action = torch.randint(0, 7, (8,))
    exps.advantage = torch.randn(8)
    exps.returnn = torch.randn(8)
    exps.log_prob = torch.randn(8)
    exps.value = torch.randn(8)

    # Create optimizer
    all_params = list(model.parameters()) + list(preprocessor.minilm_encoder.parameters())
    optimizer = torch.optim.Adam(all_params, lr=1e-4)

    # Track bert-tiny parameters
    bert_param = list(preprocessor.minilm_encoder.parameters())[0]
    initial_bert_value = bert_param.data.clone()

    print(f"\nPPO config:")
    print(f"  num_frames_per_proc: {algo.num_frames_per_proc}")
    print(f"  epochs: {algo.epochs}")
    print(f"  batch_size: {algo.batch_size}")
    print(f"  bert-tiny trainable: {not preprocessor.freeze_encoder}")

    # Run one update
    print("\nRunning PPO update...")
    try:
        logs = algo.update_parameters()
        print(f"  loss: {logs['loss']:.4f}")
        print(f"  policy_loss: {logs['policy_loss']:.4f}")
        print(f"  value_loss: {logs['value_loss']:.4f}")
        print(f"  entropy: {logs['entropy']:.4f}")
        print(f"  grad_norm: {logs['grad_norm']:.4f}")

        # Check if bert-tiny was updated
        bert_change = (bert_param.data - initial_bert_value).abs().sum().item()
        print(f"\nbert-tiny parameter change: {bert_change:.6f}")

        if bert_change > 0:
            print("✓ SUCCESS: bert-tiny is being trained during PPO!")
        else:
            print("✗ FAIL: bert-tiny parameters not changing")

        return True

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    success = test_ppo_with_fix()
    print("\n" + "=" * 60)
    print(f"Result: {'PASS' if success else 'FAIL'}")
    print("=" * 60)


if __name__ == '__main__':
    main()
