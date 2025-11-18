import copy
import numpy
import torch
import torch.nn.functional as F

from toddler_ai.algorithms.base import BaseAlgo


class PPOAlgo(BaseAlgo):
    """The class for the Proximal Policy Optimization algorithm
    ([Schulman et al., 2015](https://arxiv.org/abs/1707.06347))."""

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

        self.optimizer = torch.optim.Adam(
            self.acmodel.parameters(), lr, (beta1, beta2), eps=adam_eps
        )

        # Learning rate scheduler for stability
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
            self.target_acmodel.eval()  # Always in eval mode
            # Freeze target network parameters
            for param in self.target_acmodel.parameters():
                param.requires_grad = False
            self.target_update_freq = target_update_freq
            self.update_count = 0

        # Mixed precision training for speed and memory efficiency
        # NOTE: Only enable for CUDA; MPS has its own optimizations and may not support GradScaler
        self.mixed_precision = mixed_precision and self.device.type == 'cuda'
        self.scaler = None
        if self.mixed_precision:
            # GradScaler handles gradient scaling for mixed precision
            # Use device-agnostic API for compatibility
            self.scaler = torch.amp.GradScaler('cuda')
            print(f"Mixed precision training enabled (FP16) on {self.device}")

        self.batch_num = 0

    def collect_experiences(self):
        """Override to use target network for value bootstrapping if enabled"""
        # Call parent's collect_experiences but intercept the bootstrap value computation
        # We need to do the collection ourselves if using target network

        if not self.use_target_network:
            # Use parent implementation if not using target network
            return super().collect_experiences()

        # Custom implementation with target network for bootstrap value
        # This is based on BaseAlgo.collect_experiences() but uses target network
        for i in range(self.num_frames_per_proc):
            # Do one agent-environment interaction
            preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)

            with torch.no_grad():
                model_results = self.acmodel(preprocessed_obs, self.memory * self.mask.unsqueeze(1))
                dist = model_results['dist']
                value = model_results['value']
                memory = model_results['memory']
                extra_predictions = model_results.get('extra_predictions')

            action = dist.sample()

            # ParallelEnv returns 4 values (obs, reward, done, env_info)
            obs, reward, done, env_info = self.env.step(action.cpu().numpy())
            if self.aux_info:
                env_info = self.aux_info_collector.process(env_info)

            # Update experiences values
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

        # Add advantage and return to experiences using TARGET NETWORK for bootstrap value
        preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
        with torch.no_grad():
            # Use target network instead of main network for more stable value targets
            next_value = self.target_acmodel(preprocessed_obs, self.memory * self.mask.unsqueeze(1))['value']

        for i in reversed(range(self.num_frames_per_proc)):
            next_mask = self.masks[i+1] if i < self.num_frames_per_proc - 1 else self.mask
            next_value = self.values[i+1] if i < self.num_frames_per_proc - 1 else next_value
            next_advantage = self.advantages[i+1] if i < self.num_frames_per_proc - 1 else 0

            delta = self.rewards[i] + self.discount * next_value * next_mask - self.values[i]
            self.advantages[i] = delta + self.discount * self.gae_lambda * next_advantage * next_mask

        # Flatten the data correctly
        from toddler_ai.utils.dictlist import DictList

        exps = DictList()
        exps.obs = [self.obss[i][j]
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

        # Preprocess experiences
        exps.obs = self.preprocess_obss(exps.obs, device=self.device)

        # Log some values
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
        # Collect experiences

        exps, logs = self.collect_experiences()
        """
        exps is a DictList with the following keys ['obs', 'memory', 'mask', 'action', 'value', 'reward',
         'advantage', 'returnn', 'log_prob'] and ['collected_info', 'extra_predictions'] if we use aux_info
        exps.obs is a DictList with the following keys ['image', 'instr']
        exps.obj.image is a (n_procs * n_frames_per_proc) x image_size 4D tensor
        exps.obs.instr is a (n_procs * n_frames_per_proc) x (max number of words in an instruction) 2D tensor
        exps.memory is a (n_procs * n_frames_per_proc) x (memory_size = 2*image_embedding_size) 2D tensor
        exps.mask is (n_procs * n_frames_per_proc) x 1 2D tensor
        if we use aux_info: exps.collected_info and exps.extra_predictions are DictLists with keys
        being the added information. They are either (n_procs * n_frames_per_proc) 1D tensors or
        (n_procs * n_frames_per_proc) x k 2D tensors where k is the number of classes for multiclass classification
        """

        for _ in range(self.epochs):
            # Initialize log values

            log_entropies = []
            log_values = []
            log_policy_losses = []
            log_value_losses = []
            log_grad_norms = []

            log_losses = []

            """
            For each epoch, we create int(total_frames / batch_size + 1) batches, each of size batch_size (except
            maybe the last one. Each batch is divided into sub-batches of size recurrence (frames are contiguous in
            a sub-batch), but the position of each sub-batch in a batch and the position of each batch in the whole
            list of frames is random thanks to self._get_batches_starting_indexes().
            """

            for inds in self._get_batches_starting_indexes():
                # inds is a numpy array of indices that correspond to the beginning of a sub-batch
                # there are as many inds as there are batches
                # Initialize batch values

                batch_entropy = 0
                batch_value = 0
                batch_policy_loss = 0
                batch_value_loss = 0
                batch_loss = 0

                # Initialize memory

                memory = exps.memory[inds]

                for i in range(self.recurrence):
                    # Create a sub-batch of experience
                    sb = exps[inds + i]

                    # Normalize advantages for this sub-batch (critical for stability)
                    # This prevents exploding gradients with high reward scales
                    sb_advantage_normalized = (sb.advantage - sb.advantage.mean()) / (
                        sb.advantage.std() + 1e-8
                    )

                    # Compute loss (with mixed precision if enabled)
                    # Use autocast context manager for mixed precision forward pass
                    if self.mixed_precision:
                        with torch.amp.autocast(device_type=str(self.device.type), dtype=torch.float16):
                            model_results = self.acmodel(sb.obs, memory * sb.mask)
                    else:
                        model_results = self.acmodel(sb.obs, memory * sb.mask)

                    dist = model_results["dist"]
                    value = model_results["value"]
                    memory = model_results["memory"]
                    extra_predictions = model_results["extra_predictions"]

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

                    loss = (
                        policy_loss
                        - self.entropy_coef * entropy
                        + self.value_loss_coef * value_loss
                    )

                    # Auxiliary vision prediction loss (supplemental - helps learn better representations)
                    # This should be weak (coef ~0.01) to not interfere with RL objective
                    if (
                        "vision_pred" in extra_predictions
                        and extra_predictions["vision_pred"] is not None
                    ):
                        # Vision prediction loss: predict next observation patches
                        # We need next observation - get it from next timestep in batch
                        # For last timestep in recurrence, we don't have next obs in this sub-batch
                        if i < self.recurrence - 1:
                            sb_next = exps[inds + i + 1]
                            # Compute target vision patches from next observation
                            with torch.no_grad():
                                # Run next observation through model to get target patches
                                next_model_results = self.acmodel(
                                    sb_next.obs, memory * sb_next.mask
                                )
                                vision_target = next_model_results["extra_predictions"][
                                    "current_vision"
                                ]

                            vision_pred = extra_predictions["vision_pred"]
                            vision_pred_loss = F.mse_loss(vision_pred, vision_target)

                            # Get vision prediction coefficient from model (default 0.01 - very weak)
                            vision_coef = getattr(self.acmodel, "vision_pred_coef", 0.01)
                            loss = loss + vision_coef * vision_pred_loss

                    # Update batch values

                    batch_entropy += entropy.item()
                    batch_value += value.mean().item()
                    batch_policy_loss += policy_loss.item()
                    batch_value_loss += value_loss.item()
                    batch_loss += loss

                    # Update memories for next epoch

                    if i < self.recurrence - 1:
                        exps.memory[inds + i + 1] = memory.detach()

                # Update batch values

                batch_entropy /= self.recurrence
                batch_value /= self.recurrence
                batch_policy_loss /= self.recurrence
                batch_value_loss /= self.recurrence
                batch_loss /= self.recurrence

                # Update actor-critic

                self.optimizer.zero_grad()

                if self.mixed_precision:
                    # Mixed precision backward pass with gradient scaling
                    self.scaler.scale(batch_loss).backward()

                    # Unscale gradients for clipping
                    self.scaler.unscale_(self.optimizer)

                    # Compute gradient norm
                    grad_norm = (
                        sum(
                            p.grad.data.norm(2) ** 2
                            for p in self.acmodel.parameters()
                            if p.grad is not None
                        )
                        ** 0.5
                    )

                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(self.acmodel.parameters(), self.max_grad_norm)

                    # Optimizer step with scaled gradients
                    self.scaler.step(self.optimizer)

                    # Update scaler for next iteration
                    self.scaler.update()
                else:
                    # Standard precision training
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

                # Update log values

                log_entropies.append(batch_entropy)
                log_values.append(batch_value)
                log_policy_losses.append(batch_policy_loss)
                log_value_losses.append(batch_value_loss)
                log_grad_norms.append(grad_norm.item())
                log_losses.append(batch_loss.item())

        # Log some values

        logs["entropy"] = numpy.mean(log_entropies)
        logs["value"] = numpy.mean(log_values)
        logs["policy_loss"] = numpy.mean(log_policy_losses)
        logs["value_loss"] = numpy.mean(log_value_losses)
        logs["grad_norm"] = numpy.mean(log_grad_norms)
        logs["loss"] = numpy.mean(log_losses)

        # Step learning rate scheduler if enabled
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
            logs["lr"] = self.optimizer.param_groups[0]["lr"]

        # Update target network periodically for stable value learning
        if self.use_target_network:
            self.update_count += 1
            if self.update_count % self.target_update_freq == 0:
                # Sync target network with main network
                self.target_acmodel.load_state_dict(self.acmodel.state_dict())

        return logs

    def _get_batches_starting_indexes(self):
        """Gives, for each batch, the indexes of the observations given to
        the model and the experiences used to compute the loss at first.
        Returns
        -------
        batches_starting_indexes : list of list of int
            the indexes of the experiences to be used at first for each batch

        """

        indexes = numpy.arange(0, self.num_frames, self.recurrence)
        indexes = numpy.random.permutation(indexes)

        num_indexes = self.batch_size // self.recurrence
        batches_starting_indexes = [
            indexes[i : i + num_indexes] for i in range(0, len(indexes), num_indexes)
        ]

        return batches_starting_indexes
