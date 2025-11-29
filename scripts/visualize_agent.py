#!/usr/bin/env python3

"""
Visualize agent inference with trained models.
Uses Minigrid's built-in rendering for live visualization.
"""

import argparse
import time
import gymnasium as gym
import numpy as np
from minigrid.wrappers import RGBImgPartialObsWrapper

import toddler_ai.utils as utils
from toddler_ai.utils.agent import ModelAgent


def visualize_agent(env_name, model_name, episodes=5, delay=0.3, pixel=False):
    """
    Visualize agent playing episodes in real-time.

    Args:
        env_name: BabyAI environment name
        model_name: Trained model name
        episodes: Number of episodes to run
        delay: Delay between steps (seconds) for visualization
        pixel: Whether to use pixel observations
    """
    # Create environment with rendering
    env = gym.make(env_name, render_mode="human")
    if pixel:
        env = RGBImgPartialObsWrapper(env)

    # Load agent
    print(f"Loading model: {model_name}")
    obss_preprocessor = utils.MiniLMObssPreprocessor(model_name, env.observation_space)
    agent = ModelAgent(model_name, obss_preprocessor, argmax=True)

    print(f"\nEnvironment: {env_name}")
    print(f"Model: {model_name}")
    print(f"Episodes: {episodes}")
    print(f"Delay per step: {delay}s")
    print("\n" + "="*70)

    total_success = 0
    total_steps = 0

    for episode in range(episodes):
        obs, _ = env.reset()
        agent.on_reset()

        done = False
        episode_reward = 0
        episode_steps = 0

        print(f"\nEpisode {episode + 1}/{episodes}")
        print(f"Mission: {obs['mission']}")

        # Render initial state
        env.render()
        time.sleep(delay)

        while not done:
            # Get action from agent
            result = agent.act(obs)
            action = result['action'].item()

            # Debug: Print action and value
            print(f"  Step {episode_steps + 1}: Action={action}, Value={result['value'].item():.3f}")

            # Step environment
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            episode_steps += 1

            # Render
            env.render()
            time.sleep(delay)

            if done:
                success = reward > 0
                total_success += int(success)
                total_steps += episode_steps

                status = "SUCCESS ✓" if success else "FAILED ✗"
                print(f"  {status} - Steps: {episode_steps}, Reward: {reward:.2f}")

                # Pause longer at end of episode
                time.sleep(delay * 2)

        # Analyze feedback for memory reset
        agent.analyze_feedback(reward, done)

    env.close()

    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Success Rate: {total_success}/{episodes} ({100*total_success/episodes:.1f}%)")
    print(f"Average Steps: {total_steps/episodes:.1f}")
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize trained agent")
    parser.add_argument(
        "--env",
        default="BabyAI-GoToRedBallGrey-v0",
        help="Environment name (default: BabyAI-GoToRedBallGrey-v0)"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model name to load (REQUIRED)"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of episodes to run (default: 5)"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.3,
        help="Delay between steps in seconds (default: 0.3)"
    )
    parser.add_argument(
        "--pixel",
        action="store_true",
        help="Use pixel observations"
    )

    args = parser.parse_args()

    visualize_agent(
        env_name=args.env,
        model_name=args.model,
        episodes=args.episodes,
        delay=args.delay,
        pixel=args.pixel
    )
