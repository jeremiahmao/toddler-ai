import numpy
import torch
import torch.nn.functional as F


from toddler_ai.algorithms.base import BaseAlgo


class GRPOAlgo(BaseAlgo):
    """The class for the Group Relative Policy Optimization algorithm.
    
    GRPO uses group-based advantage estimation where advantages are computed
    relative to other samples in the same group, making it more stable for
    high-variance environments and enabling better sample efficiency.
    
    Key differences from PPO:
    - Group-based advantage normalization (advantages computed relative to group)
    - Optional KL penalty for policy regularization
    - Supports both clipped and unclipped variants
    """

    def __init__(self, envs, acmodel, num_frames_per_proc=None, discount=0.99, lr=7e-4, beta1=0.9, beta2=0.999,
                 gae_lambda=0.95, entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5, recurrence=4,
                 adam_eps=1e-5, clip_eps=0.2, epochs=4, batch_size=256, group_size=4, 
                 kl_coef=0.0, use_clipping=True, preprocess_obss=None,
                 reshape_reward=None, aux_info=None, env_seeds=None):
        """
        Parameters
        ----------
        group_size : int
            Number of samples per group for relative advantage computation.
            Advantages are normalized within each group. Must divide batch_size.
        kl_coef : float
            Coefficient for KL divergence penalty term (0.0 = no penalty).
            Helps prevent policy from deviating too far from previous policy.
        use_clipping : bool
            Whether to use PPO-style clipping. If False, uses unclipped policy gradient.
        """
        num_frames_per_proc = num_frames_per_proc or 128

        super().__init__(envs, acmodel, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                         value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward,
                         aux_info, env_seeds=env_seeds)

        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size
        self.group_size = group_size
        self.kl_coef = kl_coef
        self.use_clipping = use_clipping

        assert self.batch_size % self.recurrence == 0, "batch_size must be divisible by recurrence"
        assert self.batch_size % self.group_size == 0, "batch_size must be divisible by group_size"

        self.optimizer = torch.optim.Adam(self.acmodel.parameters(), lr, (beta1, beta2), eps=adam_eps)
        self.batch_num = 0

    def update_parameters(self):
        # Collect experiences
        exps, logs = self.collect_experiences()
        
        '''
        exps is a DictList with the following keys ['obs', 'memory', 'mask', 'action', 'value', 'reward',
         'advantage', 'returnn', 'log_prob'] and ['collected_info', 'extra_predictions'] if we use aux_info
        '''

        for epoch in range(self.epochs):
            # Initialize log values
            log_entropies = []
            log_values = []
            log_policy_losses = []
            log_value_losses = []
            log_grad_norms = []
            log_kl_divs = []
            log_losses = []

            for inds in self._get_batches_starting_indexes():
                # inds is a numpy array of indices that correspond to the beginning of a sub-batch
                
                # Initialize batch values
                batch_entropy = 0
                batch_value = 0
                batch_policy_loss = 0
                batch_value_loss = 0
                batch_kl_div = 0
                batch_loss = 0

                # Initialize memory
                memory = exps.memory[inds]

                # Collect all advantages for this batch to do group normalization
                batch_advantages = []
                
                for i in range(self.recurrence):
                    sb = exps[inds + i]
                    batch_advantages.append(sb.advantage)
                
                # Stack advantages: shape [recurrence, batch_size]
                batch_advantages = torch.stack(batch_advantages, dim=0)
                
                # Group-based advantage normalization (key GRPO feature)
                # Reshape to [recurrence, num_groups, group_size]
                num_samples = len(inds)
                num_groups = num_samples // self.group_size
                
                # Flatten recurrence and batch dims, then group
                flat_advantages = batch_advantages.reshape(-1)  # [recurrence * batch_size]
                total_samples = flat_advantages.shape[0]
                
                # Compute group statistics
                # For simplicity, we group consecutive samples
                # In practice, you might want to group by trajectory or other criteria
                grouped_advantages = flat_advantages.reshape(-1, self.group_size)  # [num_groups, group_size]
                
                # Normalize within each group (this is the core GRPO innovation)
                group_means = grouped_advantages.mean(dim=1, keepdim=True)
                group_stds = grouped_advantages.std(dim=1, keepdim=True) + 1e-8
                normalized_advantages = (grouped_advantages - group_means) / group_stds
                
                # Reshape back to [recurrence, batch_size]
                normalized_advantages = normalized_advantages.reshape(self.recurrence, num_samples)

                # Reset memory
                memory = exps.memory[inds]

                for i in range(self.recurrence):
                    # Create a sub-batch of experience
                    sb = exps[inds + i]
                    
                    # Use group-normalized advantages
                    sb_advantage_normalized = normalized_advantages[i]

                    # Compute loss
                    model_results = self.acmodel(sb.obs, memory * sb.mask)
                    dist = model_results['dist']
                    value = model_results['value']
                    memory = model_results['memory']
                    extra_predictions = model_results['extra_predictions']

                    entropy = dist.entropy().mean()

                    # Policy loss with optional clipping
                    ratio = torch.exp(dist.log_prob(sb.action) - sb.log_prob)
                    
                    if self.use_clipping:
                        # PPO-style clipped objective
                        surr1 = ratio * sb_advantage_normalized
                        surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * sb_advantage_normalized
                        policy_loss = -torch.min(surr1, surr2).mean()
                    else:
                        # Unclipped policy gradient
                        policy_loss = -(ratio * sb_advantage_normalized).mean()

                    # Optional KL penalty (helps with stability)
                    if self.kl_coef > 0:
                        # Approximate KL divergence: KL(old||new) ≈ (ratio - 1) - log(ratio)
                        kl_div = (ratio - 1.0 - torch.log(ratio)).mean()
                        batch_kl_div += kl_div.item()
                    else:
                        kl_div = torch.tensor(0.0)

                    # Value loss (clipped, same as PPO)
                    value_clipped = sb.value + torch.clamp(value - sb.value, -self.clip_eps, self.clip_eps)
                    surr1 = (value - sb.returnn).pow(2)
                    surr2 = (value_clipped - sb.returnn).pow(2)
                    value_loss = torch.max(surr1, surr2).mean()

                    # Total loss
                    loss = (policy_loss 
                            - self.entropy_coef * entropy 
                            + self.value_loss_coef * value_loss
                            + self.kl_coef * kl_div)

                    # Auxiliary vision prediction loss (if applicable)
                    if 'vision_pred' in extra_predictions and extra_predictions['vision_pred'] is not None:
                        if i < self.recurrence - 1:
                            sb_next = exps[inds + i + 1]
                            with torch.no_grad():
                                next_model_results = self.acmodel(sb_next.obs, memory * sb_next.mask)
                                vision_target = next_model_results['extra_predictions']['current_vision']

                            vision_pred = extra_predictions['vision_pred']
                            vision_pred_loss = F.mse_loss(vision_pred, vision_target)
                            vision_coef = getattr(self.acmodel, 'vision_pred_coef', 0.01)
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

                # Average batch values
                batch_entropy /= self.recurrence
                batch_value /= self.recurrence
                batch_policy_loss /= self.recurrence
                batch_value_loss /= self.recurrence
                batch_kl_div /= self.recurrence
                batch_loss /= self.recurrence

                # Update actor-critic
                self.optimizer.zero_grad()
                batch_loss.backward()
                grad_norm = sum(p.grad.data.norm(2) ** 2 for p in self.acmodel.parameters() if p.grad is not None) ** 0.5
                torch.nn.utils.clip_grad_norm_(self.acmodel.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Update log values
                log_entropies.append(batch_entropy)
                log_values.append(batch_value)
                log_policy_losses.append(batch_policy_loss)
                log_value_losses.append(batch_value_loss)
                log_grad_norms.append(grad_norm.item())
                log_kl_divs.append(batch_kl_div)
                log_losses.append(batch_loss.item())

        # Log some values
        logs["entropy"] = numpy.mean(log_entropies)
        logs["value"] = numpy.mean(log_values)
        logs["policy_loss"] = numpy.mean(log_policy_losses)
        logs["value_loss"] = numpy.mean(log_value_losses)
        logs["grad_norm"] = numpy.mean(log_grad_norms)
        logs["kl_div"] = numpy.mean(log_kl_divs)
        logs["loss"] = numpy.mean(log_losses)

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
        batches_starting_indexes = [indexes[i:i + num_indexes] for i in range(0, len(indexes), num_indexes)]

        return batches_starting_indexes