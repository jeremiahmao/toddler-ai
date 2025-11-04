#!/usr/bin/env python3

"""
Script to train the agent through reinforcment learning.
"""

import os
import logging
import csv
import json
import gymnasium as gym
import time
import datetime
import torch
import numpy as np
import subprocess

import toddler_ai
import toddler_ai.utils as utils
from toddler_ai.algorithms import PPOAlgo
from toddler_ai.utils.arguments import ArgumentParser
from toddler_ai.models.ac_model import ACModel
from toddler_ai.utils.evaluate import batch_evaluate
from toddler_ai.utils.agent import ModelAgent
from minigrid.wrappers import RGBImgPartialObsWrapper


# Parse arguments
parser = ArgumentParser()
parser.add_argument("--algo", default='ppo',
                    help="algorithm to use (default: ppo)")
parser.add_argument("--discount", type=float, default=0.99,
                    help="discount factor (default: 0.99)")
parser.add_argument("--reward-scale", type=float, default=1.0,
                    help="Reward scale multiplier (default: 1.0, optimal for sparse rewards)")
parser.add_argument("--gae-lambda", type=float, default=0.99,
                    help="lambda coefficient in GAE formula (default: 0.99, 1 means no gae)")
parser.add_argument("--value-loss-coef", type=float, default=0.25,
                    help="value loss term coefficient (default: 0.25, tuned for sparse rewards)")
parser.add_argument("--max-grad-norm", type=float, default=0.5,
                    help="maximum norm of gradient (default: 0.5)")
parser.add_argument("--clip-eps", type=float, default=0.2,
                    help="clipping epsilon for PPO (default: 0.2)")
parser.add_argument("--ppo-epochs", type=int, default=4,
                    help="number of epochs for PPO (default: 4)")
parser.add_argument("--save-interval", type=int, default=50,
                    help="number of updates between two saves (default: 50, 0 means no saving)")
args = parser.parse_args()

if __name__ == '__main__':
    utils.seed(args.seed)

    # Generate environments
    envs = []
    use_pixel = 'pixel' in args.arch
    env_seeds = []
    for i in range(args.procs):
        env = gym.make(args.env)
        if use_pixel:
            env = RGBImgPartialObsWrapper(env)
        # Store seed to use during reset (gymnasium API)
        env_seeds.append(100 * args.seed + i)
        envs.append(env)

    # Define model name
    suffix = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    instr = args.instr_arch if args.instr_arch else "noinstr"
    mem = "mem" if not args.no_mem else "nomem"
    model_name_parts = {
        'env': args.env,
        'algo': args.algo,
        'arch': args.arch,
        'instr': instr,
        'mem': mem,
        'seed': args.seed,
        'info': '',
        'coef': '',
        'suffix': suffix}
    default_model_name = "{env}_{algo}_{arch}_{instr}_{mem}_seed{seed}{info}{coef}_{suffix}".format(**model_name_parts)
    if args.pretrained_model:
        default_model_name = args.pretrained_model + '_pretrained_' + default_model_name
    args.model = args.model.format(**model_name_parts) if args.model else default_model_name

    utils.configure_logging(args.model)
    logger = logging.getLogger(__name__)

    # Define obss preprocessor
    if args.instr_arch == 'minilm':
        # Use MiniLM-based preprocessor for modern language understanding
        obss_preprocessor = utils.MiniLMObssPreprocessor(
            args.model,
            envs[0].observation_space,
            freeze_encoder=getattr(args, 'freeze_minilm', False)
        )
        logger.info('Using MiniLM language encoder (pretrained)')
    else:
        raise ValueError(f"Unsupported instr_arch: {args.instr_arch}. Only 'minilm' is supported.")

    # Define actor-critic model
    acmodel = utils.load_model(args.model, raise_not_found=False)
    if acmodel is None:
        if args.pretrained_model:
            acmodel = utils.load_model(args.pretrained_model, raise_not_found=True)
        else:
            # Use ViT model if arch='vit', otherwise use FiLM-based model
            if args.arch == 'vit':
                from toddler_ai.models.vit_model import ViTACModel
                logger.info('Using Vision Transformer (ViT) architecture')
                acmodel = ViTACModel(
                    obs_space=obss_preprocessor.obs_space,
                    action_space=envs[0].action_space,
                    image_size=7,  # BabyAI grid size
                    patch_size=1,  # Each cell is a patch
                    embed_dim=args.image_dim,
                    memory_dim=args.memory_dim,
                    use_memory=not args.no_mem,
                    vit_depth=getattr(args, 'vit_depth', 1),
                    vit_heads=getattr(args, 'vit_heads', 1),
                    cross_attn_heads=getattr(args, 'cross_attn_heads', 1),
                    dropout=getattr(args, 'dropout', 0.1)
                )
            elif args.arch == 'unified_vit':
                from toddler_ai.models.unified_vit_model import UnifiedViTACModel
                logger.info('Using Unified Concept Space ViT with Predictive Processing')
                logger.info('  - All modalities in 256-dim concept space')
                logger.info('  - Reusable action embeddings')
                logger.info('  - Working memory with temporal positions')
                logger.info('  - Vision + progress prediction')
                acmodel = UnifiedViTACModel(
                    obs_space=obss_preprocessor.obs_space,
                    action_space=envs[0].action_space,
                    image_size=7,  # BabyAI grid size
                    patch_size=1,  # Each cell is a patch
                    embed_dim=256,  # Fixed unified concept space
                    use_memory=not args.no_mem,
                    attn_depth=getattr(args, 'attn_depth', 2),
                    attn_heads=getattr(args, 'attn_heads', 4),
                    dropout=getattr(args, 'dropout', 0.1),
                    history_length=getattr(args, 'history_length', 10),
                    vision_pred_coef=getattr(args, 'vision_pred_coef', 0.1),
                    progress_pred_coef=getattr(args, 'progress_pred_coef', 0.1)
                )
            else:
                raise ValueError(f"Unsupported architecture: {args.arch}. Supported: unified_vit, vit")

    utils.save_model(acmodel, args.model)

    # Define actor-critic algo
    # Note: Algorithm decides device (may use CPU instead of MPS due to PyTorch limitations)

    reshape_reward = lambda _0, _1, reward, _2: args.reward_scale * reward
    if args.algo == "ppo":
        algo = PPOAlgo(envs, acmodel, args.frames_per_proc, args.discount, args.lr, args.beta1, args.beta2,
                       args.gae_lambda,
                       args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                       args.optim_eps, args.clip_eps, args.ppo_epochs, args.batch_size, obss_preprocessor,
                       reshape_reward, env_seeds=env_seeds)
    else:
        raise ValueError("Incorrect algorithm name: {}".format(args.algo))

    # Move model to the device chosen by the algorithm
    acmodel.to(algo.device)

    # Move MiniLM encoder to same device if using MiniLM
    if hasattr(obss_preprocessor, 'minilm_encoder') and obss_preprocessor.minilm_encoder is not None:
        obss_preprocessor.minilm_encoder.to(algo.device)
        logger.info(f'Moved MiniLM encoder to device: {algo.device}')

    # When using extra binary information, more tensors (model params) are initialized compared to when we don't use that.
    # Thus, there starts to be a difference in the random state. If we want to avoid it, in order to make sure that
    # the results of supervised-loss-coef=0. and extra-binary-info=0 match, we need to reseed here.

    utils.seed(args.seed)

    # Restore training status

    status_path = os.path.join(utils.get_log_dir(args.model), 'status.json')
    if os.path.exists(status_path):
        with open(status_path, 'r') as src:
            status = json.load(src)
    else:
        status = {'i': 0,
                  'num_episodes': 0,
                  'num_frames': 0}

    # Define logger and wandb and CSV writer

    header = (["update", "episodes", "frames", "FPS", "duration"]
              + ["return_" + stat for stat in ['mean', 'std', 'min', 'max']]
              + ["success_rate"]
              + ["num_frames_" + stat for stat in ['mean', 'std', 'min', 'max']]
              + ["entropy", "value", "policy_loss", "value_loss", "loss", "grad_norm"])

    # Initialize wandb if requested
    if args.tb:
        try:
            import wandb
            wandb.init(
                project="toddler-ai",
                name=args.model,
                config=vars(args)
            )
        except ImportError:
            logger.warning("wandb not installed. Install with: uv sync --extra tracking")
            args.tb = False
    csv_path = os.path.join(utils.get_log_dir(args.model), 'log.csv')
    first_created = not os.path.exists(csv_path)
    # we don't buffer data going in the csv log, cause we assume
    # that one update will take much longer that one write to the log
    csv_writer = csv.writer(open(csv_path, 'a', 1))
    if first_created:
        csv_writer.writerow(header)

    # Log code state, command, availability of CUDA and model

    toddler_ai_code = list(toddler_ai.__path__)[0]
    try:
        last_commit = subprocess.check_output(
            'cd {}; git log -n1'.format(toddler_ai_code), shell=True).decode('utf-8')
        logger.info('LAST COMMIT INFO:')
        logger.info(last_commit)
    except subprocess.CalledProcessError:
        logger.info('Could not figure out the last commit')
    try:
        diff = subprocess.check_output(
            'cd {}; git diff'.format(toddler_ai_code), shell=True).decode('utf-8')
        if diff:
            logger.info('GIT DIFF:')
            logger.info(diff)
    except subprocess.CalledProcessError:
        logger.info('Could not figure out the last commit')
    logger.info('COMMAND LINE ARGS:')
    logger.info(args)
    logger.info("Device: {}".format(algo.device))
    logger.info(acmodel)

    # Train model

    total_start_time = time.time()
    best_success_rate = 0
    best_mean_return = 0
    test_env_name = args.env
    while status['num_frames'] < args.frames:
        # Update parameters

        update_start_time = time.time()
        logs = algo.update_parameters()
        update_end_time = time.time()

        status['num_frames'] += logs["num_frames"]
        status['num_episodes'] += logs['episodes_done']
        status['i'] += 1

        # Print logs

        if status['i'] % args.log_interval == 0:
            total_ellapsed_time = int(time.time() - total_start_time)
            fps = logs["num_frames"] / (update_end_time - update_start_time)
            duration = datetime.timedelta(seconds=total_ellapsed_time)
            return_per_episode = utils.synthesize(logs["return_per_episode"])
            success_per_episode = utils.synthesize(
                [1 if r > 0 else 0 for r in logs["return_per_episode"]])
            num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

            data = [status['i'], status['num_episodes'], status['num_frames'],
                    fps, total_ellapsed_time,
                    *return_per_episode.values(),
                    success_per_episode['mean'],
                    *num_frames_per_episode.values(),
                    logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"],
                    logs["loss"], logs["grad_norm"]]

            format_str = ("U {} | E {} | F {:06} | FPS {:04.0f} | D {} | R:xsmM {: .2f} {: .2f} {: .2f} {: .2f} | "
                          "S {:.2f} | F:xsmM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | "
                          "pL {: .3f} | vL {:.3f} | L {:.3f} | gN {:.3f} | ")

            logger.info(format_str.format(*data))
            if args.tb:
                import wandb
                assert len(header) == len(data)
                wandb.log({key: float(value) for key, value in zip(header, data)},
                         step=status['num_frames'])

            csv_writer.writerow(data)

        # Save model

        if args.save_interval > 0 and status['i'] % args.save_interval == 0:
            with open(status_path, 'w') as dst:
                json.dump(status, dst)
                utils.save_model(acmodel, args.model)

            # Testing the model before saving
            agent = ModelAgent(args.model, obss_preprocessor, argmax=True)
            agent.model = acmodel
            agent.model.eval()
            logs = batch_evaluate(agent, test_env_name, args.val_seed, args.val_episodes, pixel=use_pixel)
            agent.model.train()
            mean_return = np.mean(logs["return_per_episode"])
            success_rate = np.mean([1 if r > 0 else 0 for r in logs['return_per_episode']])
            save_model = False
            if success_rate > best_success_rate:
                best_success_rate = success_rate
                save_model = True
            elif (success_rate == best_success_rate) and (mean_return > best_mean_return):
                best_mean_return = mean_return
                save_model = True
            if save_model:
                utils.save_model(acmodel, args.model + '_best')
                logger.info("Return {: .2f}; best model is saved".format(mean_return))
            else:
                logger.info("Return {: .2f}; not the best model; not saved".format(mean_return))
