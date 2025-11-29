import copy
import numpy
import torch
import torch.nn.functional as F

from toddler_ai.algorithms.base import BaseAlgo


class PPOBRAlgo(BaseAlgo):
    """The class for the Proximal Policy Optimization with Behavioral Reference algorithm.
    
    PPO-BR extends standard PPO by adding a behavioral cloning term that keeps the policy
    close to a reference policy. This helps with:
    - Training stability
    - Preventing catastrophic forgetting
    - Curriculum learning scenarios
    - Transfer learning from demonstrations
    
    Reference: Peng et al. "Advantage-Weighted Regression: Simple and Scalable Off-Policy RL"
    """

    def __init__(
        self,
        envs,
        acmodel,
        num_frames_per_proc=None,
        discount=0.99,
        lr=7e-4,
        beta1=0.9,
        beta2=0.999,
        gae_lambda=0.95,
        entropy_coef=0.01,
        value_loss_coef=0.5,
        max_grad_norm=0.5,
        recurrence=4,
        adam_eps=1e-5,
        clip_eps=0.2,
        epochs=4,
        batch_size=256,
        preprocess_obss=None,
        reshape_reward=None,
        aux_info=None,
        env_seeds=None,
        lr_schedule=None,
        total_frames=None,
        use_target_network=False,
        target_update_freq=10,
        mixed_precision=False,
        # PPO-BR specific parameters
        br_coef=0.1,  # Coefficient for behavioral reference loss
        reference_policy_path=None,  # Path to reference policy checkpoint
        br_decay=None,  # Optional: decay BR coefficient over time ('linear', 'exponential', None)
        br_min_coef=0.01,  # Minimum BR coefficient after decay
    ):
        num_frames_per_proc = num_frames_per_proc or 128

        super().__init__(
            envs,
            acmodel,
            num_frames_per_proc,
            discount,
            lr,
            gae_lambda,
            entropy_coef,
            value_loss_coef,
            max_grad_norm,
            recurrence,
            preprocess_obss,
            reshape_reward,
            aux_info,
            env_seeds=env_seeds,
        )

        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size

        assert self.batch_size % self.recurrence == 0

        # PPO-BR specific attributes
        self.br_coef = br_coef
        self.br_coef_initial = br_coef
        self.br_min_coef = br_min_coef
        self.br_decay = br_decay
        self.total_frames = total_frames

        # Move model and preprocessor to device
        self.acmodel.to(self.device)
        if hasattr(self.preprocess_obss, 'minilm_encoder'):
            self.preprocess_obss.minilm_encoder = self.preprocess_obss.minilm_encoder.to(self.device)
            print(f"Moved model and bert-tiny encoder to {self.device}")

        # Initialize reference policy (frozen copy of the model)
        self.reference_policy = copy.deepcopy(acmodel).to(self.device)
        self.reference_policy.eval()
        # Freeze reference policy parameters
        for param in self.reference_policy.parameters():
            param.requires_grad = False
        
        # Load reference policy from checkpoint if provided
        if reference_policy_path is not None:
            print(f"Loading reference policy from {reference_policy_path}")
            checkpoint = torch.load(reference_policy_path, map_location=self.device)
            if 'model_state' in checkpoint:
                self.reference_policy.load_state_dict(checkpoint['model_state'])
            else:
                self.reference_policy.load_state_dict(checkpoint)
            print("Reference policy loaded successfully")
        else:
            print("Using initial policy as reference (will be updated during training)")

        # Set up optimizer with bert-tiny parameters if using minilm
        if (hasattr(self.preprocess_obss, 'minilm_encoder') and
            not self.preprocess_obss.freeze_encoder):
            # Include bert-tiny parameters with differential learning rates
            minilm_lr_multiplier = 0.1
            minilm_lr = lr * minilm_lr_multiplier

            encoder_params = list(self.preprocess_obss.minilm_encoder.parameters())
            model_params = list(self.acmodel.parameters())

            param_groups = [
                {
                    'params': model_params,
                    'lr': lr,
                    'betas': (beta1, beta2),
                    'eps': adam_eps
                },
                {
                    'params': encoder_params,
                    'lr': minilm_lr,
                    'betas': (beta1, beta2),
                    'eps': adam_eps
                }
            ]

            self.optimizer = torch.optim.Adam(param_groups)
            print(f"PPO-BR optimizer includes bert-tiny: {sum(p.numel() for p in encoder_params):,} params @ LR={minilm_lr:.2e}")
            print(f"PPO-BR optimizer includes model: {sum(p.numel() for p in model_params):,} params @ LR={lr:.2e}")
        else:
            # Standard optimizer
            self.optimizer = torch.optim.Adam(
                self.acmodel.parameters(), lr, (beta1, beta2), eps=adam_eps
            )

        # Learning rate scheduler
        self.lr_scheduler = None
        if lr_schedule and total_frames:
            total_updates = total_frames // self.num_frames
            if lr_schedule == "linear":
                self.lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                    self.optimizer, start_factor=1.0, end_factor=0.1, total_iters=total_updates
                )
            elif lr_schedule == "cosine":
                self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=total_updates, eta_min=lr * 0.1
                )

        # Target network for stable value bootstrapping
        self.use_target_network = use_target_network
        if self.use_target_network:
            self.target_acmodel = copy.deepcopy(acmodel).to(self.device)
            self.target_acmodel.eval()
            for param in self.target_acmodel.parameters():
                param.requires_grad = False
            self.target_update_freq = target_update_freq
            self.update_count = 0

        # Mixed precision training
        self.mixed_precision = mixed_precision and self.device.type == 'cuda'
        self.scaler = None
        if self.mixed_precision:
            self.scaler = torch.amp.GradScaler('cuda')
            print(f"Mixed precision training enabled (FP16) on {self.device}")

        self.batch_num = 0

        # Store raw observations separately for per-batch preprocessing
        self.collected_raw_obs = None

        print(f"PPO-BR initialized with br_coef={br_coef}, br_decay={br_decay}")

    def _update_br_coef(self):
        """Update behavioral reference coefficient based on decay schedule"""
        if self.br_decay is None or self.total_frames is None:
            return
        
        progress = self.num_frames / self.total_frames
        
        if self.br_decay == "linear":
            # Linear decay from initial to min
            self.br_coef = self.br_coef_initial - progress * (self.br_coef_initial - self.br_min_coef)
        elif self.br_decay == "exponential":
            # Exponential decay
            decay_rate = numpy.log(self.br_min_coef / self.br_coef_initial)
            self.br_coef = self.br_coef_initial * numpy.exp(decay_rate * progress)
        
        self.br_coef = max(self.br_coef, self.br_min_coef)

    def collect_experiences(self):
        """Override to use target network for value bootstrapping if enabled"""
        
        if not self.use_target_network:
            # Use parent implementation but store raw observations
            exps, log = super().collect_experiences()

            # Store raw observations separately
            self.collected_raw_obs = [self.obss[i][j]
                                     for j in range(self.num_procs)
                                     for i in range(self.num_frames_per_proc)]

            # Remove preprocessed obs for fresh preprocessing
            if 'obs' in exps:
                del exps['obs']

            return exps, log

        # Custom implementation with target network
        for i in range(self.num_frames_per_proc):
            preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)

            with torch.no_grad():
                model_results = self.acmodel(preprocessed_obs, self.memory * self.mask.unsqueeze(1))
                dist = model_results['dist']
                value = model_results['value']
                extra_predictions = model_results.get('extra_predictions')

            action = dist.sample()

            # Update memory for unified_vit
            if self.args.arch == 'unified_vit' and self.acmodel.memory_size > 0:
                embed_dim = self.acmodel.embed_dim
                memory = torch.zeros_like(self.memory)
                memory[:, :-embed_dim] = self.memory[:, embed_dim:]
                memory[:, -embed_dim:] = model_results['new_memory']
                memory = memory * self.mask.unsqueeze(1)
            else:
                memory = model_results['memory']

            obs, reward, done, env_info = self.env.step(action.cpu().numpy())
            if self.aux_info:
                env_info = self.aux_info_collector.process(env_info)

            # Update experiences
            self.obss[i] = self.obs
            self.obs = obs
            self.memories[i] = self.memory
            self.memory = memory
            self.masks[i] = self.mask
            self.mask = 1 - torch.tensor(done, device=self.device, dtype=torch.float)
            self.actions[i] = action
            self.values[i] = value
            
            if self.reshape_reward is not None:
                self.rewards[i] = torch.tensor([
                    self.reshape_reward(obs_, action_, reward_, done_)
                    for obs_, action_, reward_, done_ in zip(obs, action, reward, done)
                ], device=self.device)
            else:
                self.rewards[i] = torch.tensor(reward, device=self.device)
                
            self.log_probs[i] = dist.log_prob(action)

            if self.aux_info:
                self.aux_info_collector.fill_dictionaries(i, env_info, extra_predictions)

            # Update log values
            self.log_episode_return += torch.tensor(reward, device=self.device, dtype=torch.float)
            self.log_episode_reshaped_return += self.rewards[i]
            self.log_episode_num_frames += torch.ones(self.num_procs, device=self.device)

            for idx, done_ in enumerate(done):
                if done_:
                    self.log_done_counter += 1
                    self.log_return.append(self.log_episode_return[idx].item())
                    self.log_reshaped_return.append(self.log_episode_reshaped_return[idx].item())
                    self.log_num_frames.append(self.log_episode_num_frames[idx].item())

            self.log_episode_return *= self.mask
            self.log_episode_reshaped_return *= self.mask
            self.log_episode_num_frames *= self.mask

        # Compute advantages with target network
        preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
        with torch.no_grad():
            next_value = self.target_acmodel(preprocessed_obs, self.memory * self.mask.unsqueeze(1))['value']

        for i in reversed(range(self.num_frames_per_proc)):
            next_mask = self.masks[i+1] if i < self.num_frames_per_proc - 1 else self.mask
            next_value = self.values[i+1] if i < self.num_frames_per_proc - 1 else next_value
            next_advantage = self.advantages[i+1] if i < self.num_frames_per_proc - 1 else 0

            delta = self.rewards[i] + self.discount * next_value * next_mask - self.values[i]
            self.advantages[i] = delta + self.discount * self.gae_lambda * next_advantage * next_mask

        # Flatten the data
        from toddler_ai.utils.dictlist import DictList

        exps = DictList()
        self.collected_raw_obs = [self.obss[i][j]
                                 for j in range(self.num_procs)
                                 for i in range(self.num_frames_per_proc)]

        exps.memory = self.memories.transpose(0, 1).reshape(-1, *self.memories.shape[2:])
        exps.mask = self.masks.transpose(0, 1).reshape(-1).unsqueeze(1)
        exps.action = self.actions.transpose(0, 1).reshape(-1)
        exps.value = self.values.transpose(0, 1).reshape(-1)
        exps.reward = self.rewards.transpose(0, 1).reshape(-1)
        exps.advantage = self.advantages.transpose(0, 1).reshape(-1)
        exps.returnn = exps.value + exps.advantage
        exps.log_prob = self.log_probs.transpose(0, 1).reshape(-1)

        if self.aux_info:
            exps = self.aux_info_collector.end_collection(exps)

        if 'obs' in exps:
            del exps['obs']

        # Log values
        keep = max(self.log_done_counter, self.num_procs)

        log = {
            "return_per_episode": self.log_return[-keep:],
            "reshaped_return_per_episode": self.log_reshaped_return[-keep:],
            "num_frames_per_episode": self.log_num_frames[-keep:],
            "num_frames": self.num_frames,
            "episodes_done": self.log_done_counter,
        }

        self.log_done_counter = 0
        self.log_return = self.log_return[-self.num_procs:]
        self.log_reshaped_return = self.log_reshaped_return[-self.num_procs:]
        self.log_num_frames = self.log_num_frames[-self.num_procs:]

        return exps, log

    def update_parameters(self):
        """Update parameters with PPO-BR objective"""
        
        # Update BR coefficient based on decay schedule
        self._update_br_coef()
        
        # Collect experiences
        exps, logs = self.collect_experiences()

        for _ in range(self.epochs):
            # Initialize log values
            log_entropies = []
            log_values = []
            log_policy_losses = []
            log_value_losses = []
            log_br_losses = []  # Track behavioral reference loss
            log_grad_norms = []
            log_losses = []

            for inds in self._get_batches_starting_indexes():
                # Preprocess observations fresh for this batch
                batch_indices = []
                for i in range(self.recurrence):
                    batch_indices.extend((inds + i).tolist())

                if self.collected_raw_obs is None:
                    raise RuntimeError("collected_raw_obs is None - collect_experiences() must be called first")

                batch_raw_obs = [self.collected_raw_obs[idx] for idx in batch_indices]
                batch_obs = self.preprocess_obss(batch_raw_obs, device=self.device)

                # Initialize batch values
                batch_entropy = 0
                batch_value = 0
                batch_policy_loss = 0
                batch_value_loss = 0
                batch_br_loss = 0  # Behavioral reference loss
                batch_loss = 0

                # Initialize memory
                memory = exps.memory[inds]

                for i in range(self.recurrence):
                    # Create a sub-batch of experience
                    sb = exps[inds + i]

                    # Get batch-local observations for this sub-batch
                    sb_obs_start = i * len(inds)
                    sb_obs_end = (i + 1) * len(inds)
                    sb_obs = batch_obs[sb_obs_start:sb_obs_end]

                    # Normalize advantages
                    sb_advantage_normalized = (sb.advantage - sb.advantage.mean()) / (
                        sb.advantage.std() + 1e-8
                    )

                    # Forward pass through current policy
                    if self.mixed_precision:
                        with torch.amp.autocast(device_type=str(self.device.type), dtype=torch.float16):
                            model_results = self.acmodel(sb_obs, memory * sb.mask)
                    else:
                        model_results = self.acmodel(sb_obs, memory * sb.mask)

                    dist = model_results["dist"]
                    value = model_results["value"]
                    memory = model_results["memory"]
                    extra_predictions = model_results["extra_predictions"]

                    # Compute standard PPO losses
                    entropy = dist.entropy().mean()

                    ratio = torch.exp(dist.log_prob(sb.action) - sb.log_prob)
                    surr1 = ratio * sb_advantage_normalized
                    surr2 = (
                        torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
                        * sb_advantage_normalized
                    )
                    policy_loss = -torch.min(surr1, surr2).mean()

                    value_clipped = sb.value + torch.clamp(
                        value - sb.value, -self.clip_eps, self.clip_eps
                    )
                    surr1 = (value - sb.returnn).pow(2)
                    surr2 = (value_clipped - sb.returnn).pow(2)
                    value_loss = torch.max(surr1, surr2).mean()

                    # Compute behavioral reference loss (KL divergence to reference policy)
                    with torch.no_grad():
                        if self.mixed_precision:
                            with torch.amp.autocast(device_type=str(self.device.type), dtype=torch.float16):
                                ref_results = self.reference_policy(sb_obs, memory * sb.mask)
                        else:
                            ref_results = self.reference_policy(sb_obs, memory * sb.mask)
                        ref_dist = ref_results["dist"]
                    
                    # KL divergence from current policy to reference policy
                    # For categorical distributions: KL(ref || current)
                    br_loss = torch.distributions.kl_divergence(ref_dist, dist).mean()

                    # Total loss with behavioral reference term
                    loss = (
                        policy_loss
                        - self.entropy_coef * entropy
                        + self.value_loss_coef * value_loss
                        + self.br_coef * br_loss  # Add BR regularization
                    )

                    # Auxiliary vision prediction loss
                    if (
                        "vision_pred" in extra_predictions
                        and extra_predictions["vision_pred"] is not None
                    ):
                        if i < self.recurrence - 1:
                            sb_next = exps[inds + i + 1]
                            sb_next_obs_start = (i + 1) * len(inds)
                            sb_next_obs_end = (i + 2) * len(inds)
                            sb_next_obs = batch_obs[sb_next_obs_start:sb_next_obs_end]
                            
                            with torch.no_grad():
                                next_model_results = self.acmodel(
                                    sb_next_obs, memory * sb_next.mask
                                )
                                vision_target = next_model_results["extra_predictions"][
                                    "current_vision"
                                ]

                            vision_pred = extra_predictions["vision_pred"]
                            vision_pred_loss = F.mse_loss(vision_pred, vision_target)
                            vision_coef = getattr(self.acmodel, "vision_pred_coef", 0.01)
                            loss = loss + vision_coef * vision_pred_loss

                    # Update batch values
                    batch_entropy += entropy.item()
                    batch_value += value.mean().item()
                    batch_policy_loss += policy_loss.item()
                    batch_value_loss += value_loss.item()
                    batch_br_loss += br_loss.item()
                    batch_loss += loss

                    # Update memories for next epoch
                    if i < self.recurrence - 1:
                        exps.memory[inds + i + 1] = memory.detach()

                # Average batch values
                batch_entropy /= self.recurrence
                batch_value /= self.recurrence
                batch_policy_loss /= self.recurrence
                batch_value_loss /= self.recurrence
                batch_br_loss /= self.recurrence
                batch_loss /= self.recurrence

                # Backward pass and optimization
                self.optimizer.zero_grad()

                if self.mixed_precision:
                    self.scaler.scale(batch_loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    
                    grad_norm = (
                        sum(
                            p.grad.data.norm(2) ** 2
                            for p in self.acmodel.parameters()
                            if p.grad is not None
                        )
                        ** 0.5
                    )
                    
                    torch.nn.utils.clip_grad_norm_(self.acmodel.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    batch_loss.backward()
                    
                    grad_norm = (
                        sum(
                            p.grad.data.norm(2) ** 2
                            for p in self.acmodel.parameters()
                            if p.grad is not None
                        )
                        ** 0.5
                    )
                    
                    torch.nn.utils.clip_grad_norm_(self.acmodel.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                # Convert batch_loss to item for logging
                batch_loss = batch_loss.item()

                # Update log values
                log_entropies.append(batch_entropy)
                log_values.append(batch_value)
                log_policy_losses.append(batch_policy_loss)
                log_value_losses.append(batch_value_loss)
                log_br_losses.append(batch_br_loss)
                log_grad_norms.append(grad_norm.item())
                log_losses.append(batch_loss)

        # Log aggregated values
        logs["entropy"] = numpy.mean(log_entropies)
        logs["value"] = numpy.mean(log_values)
        logs["policy_loss"] = numpy.mean(log_policy_losses)
        logs["value_loss"] = numpy.mean(log_value_losses)
        logs["br_loss"] = numpy.mean(log_br_losses)  # Log BR loss
        logs["br_coef"] = self.br_coef  # Log current BR coefficient
        logs["grad_norm"] = numpy.mean(log_grad_norms)
        logs["loss"] = numpy.mean(log_losses)

        # Step learning rate scheduler
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
            logs["lr"] = self.optimizer.param_groups[0]["lr"]

        # Update target network periodically
        if self.use_target_network:
            self.update_count += 1
            if self.update_count % self.target_update_freq == 0:
                self.target_acmodel.load_state_dict(self.acmodel.state_dict())

        return logs

    def _get_batches_starting_indexes(self):
        """Generate batch starting indexes for training"""
        indexes = numpy.arange(0, self.num_frames, self.recurrence)
        indexes = numpy.random.permutation(indexes)

        num_indexes = self.batch_size // self.recurrence
        batches_starting_indexes = [
            indexes[i : i + num_indexes] for i in range(0, len(indexes), num_indexes)
        ]

        return batches_starting_indexes
    
    def update_reference_policy(self):
        """Manually update the reference policy to the current policy.
        
        This can be called periodically during training to update the reference
        for curriculum learning or continual learning scenarios.
        """
        self.reference_policy.load_state_dict(self.acmodel.state_dict())
        print("Reference policy updated to current policy")