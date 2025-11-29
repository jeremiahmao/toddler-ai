#!/usr/bin/env python3
"""
Visualize how bert-tiny word embeddings changed during training.

This script:
1. Loads the pretrained bert-tiny encoder from the model
2. Extracts word embeddings for task-relevant vocabulary
3. Compares with the original pretrained bert-tiny
4. Visualizes changes using t-SNE and PCA
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from transformers import AutoModel, AutoTokenizer

import toddler_ai.utils as utils


def load_model_bert_encoder(model_name):
    """Load the bert-tiny encoder from a trained model."""
    model_path = Path("models") / model_name

    # Load the preprocessor which contains the encoder
    env_name = "BabyAI-GoToRedBallGrey-v0"  # Dummy env for loading
    import gymnasium as gym
    env = gym.make(env_name)

    obss_preprocessor = utils.MiniLMObssPreprocessor(model_name, env.observation_space)
    env.close()

    return obss_preprocessor.minilm_encoder, obss_preprocessor.tokenizer


def get_word_embeddings(encoder, tokenizer, words):
    """Extract word embeddings from bert-tiny encoder.

    Returns both token embeddings and CLS-pooled sentence embeddings.
    """
    # Get token embeddings (first layer of BERT)
    token_embeddings_layer = encoder.embeddings.word_embeddings

    # Tokenize words
    token_ids = []
    for word in words:
        tokens = tokenizer(word, add_special_tokens=False)
        # Get first token if word is split into multiple tokens
        if len(tokens['input_ids']) > 0:
            token_ids.append(tokens['input_ids'][0])
        else:
            token_ids.append(tokenizer.unk_token_id)

    # Get embeddings
    token_ids_tensor = torch.tensor(token_ids)
    token_embeds = token_embeddings_layer(token_ids_tensor).detach().numpy()

    # Also get full sentence embeddings (CLS token after full forward pass)
    with torch.no_grad():
        encoded = tokenizer(words, padding=True, truncation=True, return_tensors='pt')
        outputs = encoder(**encoded)
        sentence_embeds = outputs.last_hidden_state[:, 0, :].numpy()  # CLS token

    return token_embeds, sentence_embeds


def main():
    parser = argparse.ArgumentParser(description="Visualize bert-tiny embedding changes")
    parser.add_argument(
        "--model",
        required=True,
        help="Trained model name (e.g., em_p3_GoToObj_ppo)"
    )
    parser.add_argument(
        "--method",
        default="tsne",
        choices=["tsne", "pca", "both"],
        help="Dimensionality reduction method"
    )
    parser.add_argument(
        "--output",
        default="bert_embeddings.png",
        help="Output filename for visualization"
    )

    args = parser.parse_args()

    # Define task-relevant vocabulary
    # These are the key words that appear in BabyAI instructions
    task_words = [
        # Actions
        "go", "pick", "put", "open",
        # Objects
        "ball", "key", "box", "door",
        # Colors
        "red", "green", "blue", "purple", "yellow", "grey",
        # Spatial
        "left", "right", "front", "behind",
        # Demonstratives
        "the", "a", "to",
    ]

    print("=" * 70)
    print("BERT-tiny Embedding Visualization")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Vocabulary size: {len(task_words)} words")
    print()

    # Load original pretrained bert-tiny
    print("Loading original pretrained bert-tiny...")
    original_encoder = AutoModel.from_pretrained('prajjwal1/bert-tiny')
    original_tokenizer = AutoTokenizer.from_pretrained('prajjwal1/bert-tiny')

    # Load trained model's bert-tiny
    print(f"Loading bert-tiny from trained model: {args.model}")
    trained_encoder, trained_tokenizer = load_model_bert_encoder(args.model)

    # Extract embeddings
    print("\nExtracting embeddings...")
    orig_token_embeds, orig_sentence_embeds = get_word_embeddings(
        original_encoder, original_tokenizer, task_words
    )
    trained_token_embeds, trained_sentence_embeds = get_word_embeddings(
        trained_encoder, trained_tokenizer, task_words
    )

    # Compute embedding changes
    token_deltas = trained_token_embeds - orig_token_embeds
    sentence_deltas = trained_sentence_embeds - orig_sentence_embeds

    token_norms = np.linalg.norm(token_deltas, axis=1)
    sentence_norms = np.linalg.norm(sentence_deltas, axis=1)

    print("\nEmbedding Changes:")
    print(f"  Token embeddings - Mean change: {token_norms.mean():.4f}, Max: {token_norms.max():.4f}")
    print(f"  Sentence embeddings - Mean change: {sentence_norms.mean():.4f}, Max: {sentence_norms.max():.4f}")

    # Print top changed words
    print("\nTop 10 most changed words (sentence embeddings):")
    top_indices = np.argsort(sentence_norms)[::-1][:10]
    for i, idx in enumerate(top_indices, 1):
        print(f"  {i}. '{task_words[idx]}': {sentence_norms[idx]:.4f}")

    # Visualization
    print(f"\nCreating visualization using {args.method}...")

    # Combine embeddings for visualization
    all_embeds = np.vstack([orig_sentence_embeds, trained_sentence_embeds])

    if args.method == "tsne" or args.method == "both":
        # t-SNE visualization
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(5, len(task_words)-1))
        embeds_2d = tsne.fit_transform(all_embeds)

        orig_2d = embeds_2d[:len(task_words)]
        trained_2d = embeds_2d[len(task_words):]

        if args.method == "both":
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            ax = ax1
        else:
            fig, ax = plt.subplots(figsize=(10, 8))

        # Plot arrows showing embedding shifts
        for i, word in enumerate(task_words):
            # Draw arrow from original to trained
            ax.arrow(orig_2d[i, 0], orig_2d[i, 1],
                    trained_2d[i, 0] - orig_2d[i, 0],
                    trained_2d[i, 1] - orig_2d[i, 1],
                    head_width=0.3, head_length=0.3, fc='gray', ec='gray', alpha=0.3)

            # Plot points
            ax.scatter(orig_2d[i, 0], orig_2d[i, 1], c='blue', s=100, alpha=0.6, label='Original' if i == 0 else '')
            ax.scatter(trained_2d[i, 0], trained_2d[i, 1], c='red', s=100, alpha=0.6, label='Trained' if i == 0 else '')

            # Annotate
            ax.text(trained_2d[i, 0], trained_2d[i, 1], word, fontsize=9, alpha=0.8)

        ax.set_title(f't-SNE: BERT Embeddings Before/After Training\nModel: {args.model}')
        ax.set_xlabel('t-SNE Component 1')
        ax.set_ylabel('t-SNE Component 2')
        ax.legend()
        ax.grid(True, alpha=0.3)

    if args.method == "pca" or args.method == "both":
        # PCA visualization
        pca = PCA(n_components=2)
        embeds_2d = pca.fit_transform(all_embeds)

        orig_2d = embeds_2d[:len(task_words)]
        trained_2d = embeds_2d[len(task_words):]

        if args.method == "both":
            ax = ax2
        else:
            fig, ax = plt.subplots(figsize=(10, 8))

        # Plot arrows showing embedding shifts
        for i, word in enumerate(task_words):
            ax.arrow(orig_2d[i, 0], orig_2d[i, 1],
                    trained_2d[i, 0] - orig_2d[i, 0],
                    trained_2d[i, 1] - orig_2d[i, 1],
                    head_width=0.3, head_length=0.3, fc='gray', ec='gray', alpha=0.3)

            ax.scatter(orig_2d[i, 0], orig_2d[i, 1], c='blue', s=100, alpha=0.6, label='Original' if i == 0 else '')
            ax.scatter(trained_2d[i, 0], trained_2d[i, 1], c='red', s=100, alpha=0.6, label='Trained' if i == 0 else '')

            ax.text(trained_2d[i, 0], trained_2d[i, 1], word, fontsize=9, alpha=0.8)

        var_explained = pca.explained_variance_ratio_
        ax.set_title(f'PCA: BERT Embeddings Before/After Training\nModel: {args.model}\n'
                    f'Variance explained: PC1={var_explained[0]:.1%}, PC2={var_explained[1]:.1%}')
        ax.set_xlabel(f'PC1 ({var_explained[0]:.1%} variance)')
        ax.set_ylabel(f'PC2 ({var_explained[1]:.1%} variance)')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {args.output}")
    print("=" * 70)


if __name__ == "__main__":
    main()
