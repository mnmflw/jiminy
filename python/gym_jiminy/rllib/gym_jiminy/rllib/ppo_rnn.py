"""Implement several regularization losses on top of the original PPO algorithm
to encourage smoothness of the action and clustering of the behavior of the
policy without having to rework the reward function itself. It takes advantage
of the analytical gradient of the policy.
"""
import math
import operator
from functools import partial, reduce
from typing import Any, Dict, List, Optional, Tuple, Type, Union, cast

import gymnasium as gym
import torch
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.algorithms.ppo import PPO as _PPO
from ray.rllib.algorithms.ppo import PPOConfig as _PPOConfig
from ray.rllib.algorithms.ppo.ppo_tf_policy import validate_config
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy as _PPOTorchPolicy
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_action_dist import TorchDiagGaussian
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.torch_mixins import (
    EntropyCoeffSchedule,
    KLCoeffMixin,
    LearningRateSchedule,
    ValueNetworkMixin,
)
from ray.rllib.policy.torch_policy_v2 import TorchPolicyV2
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils.annotations import override
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.torch_utils import (
    explained_variance,
    l2_loss,
    sequence_mask,
    warn_if_infinite_kl_divergence,
)
from ray.rllib.utils.typing import TensorType


def get_action_mean(model: ModelV2,
                    dist_class: Union[partial, Type[ActionDistribution]],
                    action_logits: torch.Tensor) -> torch.Tensor:
    """Compute the mean value of the actions based on action distribution
    logits and type of distribution.

    .. note:
        It performs deterministic sampling for all distributions except
        multivariate independent normal distribution, for which the mean can be
        very efficiently extracted as a view of the logits.
    """
    # Extract wrapped distribution class
    dist_class_unwrapped: Type[ActionDistribution]
    if isinstance(dist_class, partial):
        dist_class_func = cast(Type[ActionDistribution], dist_class.func)
        assert issubclass(dist_class_func, ActionDistribution)
        dist_class_unwrapped = dist_class_func
    else:
        dist_class_unwrapped = dist_class

    # Efficient specialization for `TorchDiagGaussian` distribution
    if issubclass(dist_class_unwrapped, TorchDiagGaussian):
        action_mean, _ = torch.chunk(action_logits, 2, dim=1)
        return action_mean

    # Slow but generic fallback
    action_dist = dist_class(action_logits, model)
    return action_dist.deterministic_sample()


def get_adversarial_observation_sgld(
        model: ModelV2,
        dist_class: Type[ActionDistribution],
        train_batch: SampleBatch,
        noise_scale: float,
        beta_inv: float,
        n_steps: int,
        action_true_mean: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
    """Compute adversarial observation maximizing Mean Squared Error between
    the original and the perturbed mean action using Stochastic gradient
    Langevin dynamics algorithm (SGLD).
    """
    # Compute mean field action for true observation if not provided
    if action_true_mean is None:
        with torch.no_grad():
            action_true_logits, _ = model(train_batch)
            action_true_mean = get_action_mean(
                model, dist_class, action_true_logits)
    else:
        action_true_mean = action_true_mean.detach()

    # Shallow copy the original training batch.
    # Be careful accessing fields using the original batch to properly keep
    # track of accessed keys, which will be used to automatically discard
    # useless components of policy's view requirements.
    train_batch_copy = train_batch.copy(shallow=True)

    # Extract original observation
    observation_true = train_batch["obs"]

    # Define observation upper and lower bounds for clipping
    obs_lb_flat = observation_true - noise_scale
    obs_ub_flat = observation_true + noise_scale

    # Adjust the step size based on noise scale and number of steps
    step_eps = noise_scale / n_steps

    # Use Stochastic gradient Langevin dynamics (SGLD) to compute adversary
    # observation perturbation. It consists in find nearby observations that
    # maximize the mean action difference.
    observation_noisy = observation_true + step_eps * 2.0 * (
        torch.empty_like(observation_true).bernoulli_(p=0.5) - 0.5)
    for i in range(n_steps):
        with torch.torch.enable_grad():
            # Make sure gradient computation is required
            observation_noisy.requires_grad_(True)

            # Compute mean field action for noisy observation
            train_batch_copy["obs"] = observation_noisy
            action_noisy_logits, _ = model(train_batch_copy)
            action_noisy_mean = get_action_mean(
                model, dist_class, action_noisy_logits)

            # Compute action different and associated gradient
            loss = torch.mean(torch.sum(
                (action_noisy_mean - action_true_mean) ** 2, dim=-1))
            loss.backward()

        # compute the noisy gradient for observation update
        noise_factor = math.sqrt(2.0 * step_eps * beta_inv) / (i + 2)
        observation_update = observation_noisy.grad + \
            noise_factor * torch.randn_like(observation_true)

        # Need to clear gradients before the backward() for policy_loss
        observation_noisy.detach_()

        # Project gradient to step boundary.
        # Note that `sign` is used to be agnostic to the norm of the gradient,
        # which would require to tune the learning rate or use an adaptive step
        # method. Alternatively, the normalized gradient could be used, but it
        # takes more iterations to converge in practice.
        # TODO: The update step should be `step_eps` but it was found that
        # using `noise_scale` converges faster.
        observation_noisy += observation_update.sign() * noise_scale

        # clip into the upper and lower bounds
        observation_noisy.clamp_(obs_lb_flat, obs_ub_flat)

    return observation_noisy


def _compute_mirrored_value(value: torch.Tensor,
                            space: gym.spaces.Box,
                            mirror_mat: Union[
                                Dict[str, torch.Tensor], torch.Tensor]
                            ) -> torch.Tensor:
    """Compute mirrored value from observation space based on provided
    mirroring transformation.
    """
    def _update_flattened_slice(data: torch.Tensor,
                                shape: Tuple[int, ...],
                                mirror_mat: torch.Tensor) -> torch.Tensor:
        """Mirror an array of flattened tensor using provided transformation
        matrix.
        """
        mirror_mat = mirror_mat.to(data.device)
        if len(shape) > 1:
            assert len(shape) == 2, "shape > 2 is not supported for now."
            data = data.reshape(
                (-1, *shape)).swapaxes(1, 0).reshape((shape[0], -1))
            data_mirrored = mirror_mat @ data
            return data_mirrored.reshape(
                (shape[0], -1, shape[1])).swapaxes(1, 0).reshape((-1, *shape))
        return torch.mm(data, mirror_mat)

    if isinstance(mirror_mat, dict):
        offset = 0
        value_mirrored = []
        for field, slice_mirror_mat in mirror_mat.items():
            field_shape = space.original_space[  # type: ignore[attr-defined]
                field].shape
            field_size = reduce(operator.mul, field_shape)
            field_slice = slice(offset, offset + field_size)
            slice_mirrored = _update_flattened_slice(
                value[:, field_slice], field_shape, slice_mirror_mat)
            value_mirrored.append(slice_mirrored)
            offset += field_size
        return torch.cat(value_mirrored, dim=1)
    return _update_flattened_slice(value, space.shape, mirror_mat)


class PPOConfig(_PPOConfig):
    """Provide additional parameters on top of the original PPO algorithm to
    configure several regularization losses. See `PPOTorchPolicy` for details.
    """
    def __init__(self, algo_class: Optional[Type["PPO"]] = None):
        super().__init__(algo_class=algo_class or PPO)

        self.spatial_noise_scale = 1.0
        self.enable_adversarial_noise = False
        self.sgld_beta_inv = 1e-8
        self.sgld_n_steps = 10
        self.temporal_barrier_scale = 10.0
        self.temporal_barrier_threshold = float('inf')
        self.temporal_barrier_reg = 0.0
        self.symmetric_policy_reg = 0.0
        self.enable_symmetry_surrogate_loss = False
        self.caps_temporal_reg = 0.0
        self.caps_spatial_reg = 0.0
        self.caps_global_reg = 0.0
        self.l2_reg = 0.0

    @override(_PPOConfig)
    def training(
        self,
        *,
        enable_adversarial_noise: Optional[bool] = None,
        spatial_noise_scale: Optional[float] = None,
        sgld_beta_inv: Optional[float] = None,
        sgld_n_steps: Optional[int] = None,
        temporal_barrier_scale: Optional[float] = None,
        temporal_barrier_threshold: Optional[float] = None,
        temporal_barrier_reg: Optional[float] = None,
        symmetric_policy_reg: Optional[float] = None,
        enable_symmetry_surrogate_loss: Optional[bool] = None,
        caps_temporal_reg: Optional[float] = None,
        caps_spatial_reg: Optional[float] = None,
        caps_global_reg: Optional[float] = None,
        l2_reg: Optional[float] = None,
        **kwargs: Any,
    ) -> "PPOConfig":
        # Pass kwargs onto super's `training()` method.
        super().training(**kwargs)

        if enable_adversarial_noise is not None:
            self.enable_adversarial_noise = enable_adversarial_noise
        if spatial_noise_scale is not None:
            self.spatial_noise_scale = spatial_noise_scale
        if sgld_beta_inv is not None:
            self.sgld_beta_inv = sgld_beta_inv
        if sgld_n_steps is not None:
            self.sgld_n_steps = sgld_n_steps
        if temporal_barrier_scale is not None:
            self.temporal_barrier_scale = temporal_barrier_scale
        if temporal_barrier_threshold is not None:
            self.temporal_barrier_threshold = temporal_barrier_threshold
        if temporal_barrier_reg is not None:
            self.temporal_barrier_reg = temporal_barrier_reg
        if symmetric_policy_reg is not None:
            self.symmetric_policy_reg = symmetric_policy_reg
        if enable_symmetry_surrogate_loss is not None:
            self.enable_symmetry_surrogate_loss = \
                enable_symmetry_surrogate_loss
        if caps_temporal_reg is not None:
            self.caps_temporal_reg = caps_temporal_reg
        if caps_spatial_reg is not None:
            self.caps_spatial_reg = caps_spatial_reg
        if caps_global_reg is not None:
            self.caps_global_reg = caps_global_reg
        if l2_reg is not None:
            self.l2_reg = l2_reg

        return self


class PPO(_PPO):
    """Custom PPO algorithm with additional regularization losses on top of the
    original surrogate loss. See `PPOTorchPolicy` for details.
    """
    @classmethod
    @override(_PPO)
    def get_default_config(cls) -> AlgorithmConfig:
        """Returns a default configuration for the algorithm.
        """
        return PPOConfig()

    @classmethod
    @override(_PPO)
    def get_default_policy_class(cls,
                                 config: AlgorithmConfig
                                 ) -> Optional[Type[Policy]]:
        """Returns a default Policy class to use, given a config.
        """
        framework = config.framework_str
        if framework == "torch":
            return PPOTorchPolicy
        raise ValueError(f"The framework {framework} is not supported.")


class PPOTorchPolicy(_PPOTorchPolicy):
    """Add regularization losses on top of the original loss of PPO.

    More specifically, it adds:
        - CAPS regularization, which combines the spatial and temporal
        difference between previous and current state.
        - Global regularization, which is the average norm of the action
        - temporal barrier, which is exponential barrier loss when the
        normalized action is above a threshold (much like interior point
        methods).
        - symmetry regularization, which is the error between actions and
        symmetric actions associated with symmetric observations.
        - symmetry surrogate loss, which is the surrogate loss associated
        with the symmetric (actions, observations) spaces. As the surrogate
        loss goal is to increase the likelihood of selecting higher reward
        actions given the current state, the symmetry surrogate loss enables
        equivalent likelihood increase for selecting the symmetric higher
        reward actions given the symmetric state.
        - L2 regularization of policy network weights.

    More insights on the regularization losses with their emerging properties,
    and on how to tune the parameters can be found in the reference articles:
        - A. Duburcq, F. Schramm, G. Boeris, N. Bredeche, and Y. Chevaleyre,
        “Reactive Stepping for Humanoid Robots using Reinforcement Learning:
        Application to Standing Push Recovery on the Exoskeleton Atalante,” in
        International Conference on Intelligent Robots and Systems (IROS),
        vol. 2022-Octob. IEEE, oct 2022, pp. 9302–9309
        - S. Mysore, B. Mabsout, R. Mancuso, and K. Saenko, “Regularizing
        action policies for smooth control with reinforcement learning,”
        IEEE International Conference on Robotics and Automation (ICRA),
        pp. 1810–1816, 2021
        - M. Mittal, N. Rudin, V. Klemm, A. Allshire, and M. Hutter,
        “Symmetry considerations for learning task symmetric robot policies,”
        arXiv preprint arXiv:2403.04359, 2024.
    """
    def __init__(self,
                 observation_space: gym.spaces.Space,
                 action_space: gym.spaces.Space,
                 config: Union[PPOConfig, Dict[str, Any]]) -> None:
        """Initialize PPO Torch policy.

        It extracts observation mirroring transforms for symmetry computations.
        """
        # pylint: disable=non-parent-init-called,super-init-not-called

        # Convert any type of input dict input classical dictionary for compat
        config_dict: Dict[str, Any] = {**PPOConfig().to_dict(), **config}
        validate_config(config_dict)

        # Call base implementation. Note that `PPOTorchPolicy.__init__` is
        # bypassed because it calls `_initialize_loss_from_dummy_batch`
        # automatically, and mirroring matrices are not extracted at this
        # point. It is not possible to extract them since `self.device` is set
        # by `TorchPolicyV2.__init__`.
        TorchPolicyV2.__init__(
            self,
            observation_space,
            action_space,
            config_dict,
            max_seq_len=config_dict["model"]["max_seq_len"],
        )

        # Initialize mixins
        ValueNetworkMixin.__init__(self, config_dict)
        LearningRateSchedule.__init__(
            self, config_dict["lr"], config_dict["lr_schedule"])
        EntropyCoeffSchedule.__init__(self,
                                      config_dict["entropy_coeff"],
                                      config_dict["entropy_coeff_schedule"])
        KLCoeffMixin.__init__(self, config_dict)

        # Extract and convert observation and action mirroring transform
        self.obs_mirror_mat: Optional[Union[
            Dict[str, torch.Tensor], torch.Tensor]] = None
        self.action_mirror_mat: Optional[Union[
            Dict[str, torch.Tensor], torch.Tensor]] = None
        if config_dict["symmetric_policy_reg"] > 0.0 or \
                config_dict["enable_symmetry_surrogate_loss"]:
            # Observation space
            is_obs_dict = hasattr(observation_space, "original_space")
            if is_obs_dict:
                observation_space = observation_space.\
                    original_space  # type: ignore[attr-defined]
                self.obs_mirror_mat = {}
                for field, mirror_mat in observation_space.\
                        mirror_mat.items():  # type: ignore[attr-defined]
                    obs_mirror_mat = torch.tensor(mirror_mat,
                                                  dtype=torch.float32,
                                                  device=self.device)
                    self.obs_mirror_mat[field] = obs_mirror_mat.T.contiguous()
            else:
                obs_mirror_mat = torch.tensor(
                    observation_space.mirror_mat,  # type: ignore[attr-defined]
                    dtype=torch.float32,
                    device=self.device)
                self.obs_mirror_mat = obs_mirror_mat.T.contiguous()

            # Action space
            action_mirror_mat = torch.tensor(
                action_space.mirror_mat,  # type: ignore[attr-defined]
                dtype=torch.float32,
                device=self.device)
            self.action_mirror_mat = action_mirror_mat.T.contiguous()

        self._initialize_loss_from_dummy_batch()
        self.config: Dict[str, Any]

    def _get_default_view_requirements(self) -> None:
        """Add previous observation to view requirements for CAPS
        regularization.
        """
        view_requirements = super()._get_default_view_requirements()
        view_requirements["prev_obs"] = ViewRequirement(
            data_col=SampleBatch.OBS,
            space=self.observation_space,
            shift=-1,
            batch_repeat_value=1,
            used_for_compute_actions=False,
            used_for_training=True)
        return view_requirements

    @override(_PPOTorchPolicy)
    def loss(
        self,
        model: ModelV2,
        dist_class: Type[ActionDistribution],
        train_batch: SampleBatch,
    ) -> Union[TensorType, List[TensorType]]:
        """Compute loss for Proximal Policy Objective with additional regularizations.

        Args:
            model: The Model to calculate the loss for.
            dist_class: The action distr. class.
            train_batch: The training data.

        Returns:
            The PPO loss tensor given the input batch.
        """
        with torch.no_grad():
            # Extract some proxies from convenience
            observation_true = train_batch[SampleBatch.OBS]

            # Initialize the various training batches to forward to the model
            train_batches = {"true": train_batch}


            if self.config["caps_spatial_reg"] > 0.0:
                # Shallow copy the original training batch
                train_batch_noisy = train_batch.copy(shallow=True)

                # Generate noisy observation
                observation_noisy = torch.normal(
                    observation_true, self.config["spatial_noise_scale"])

                # Replace current observation by the noisy one
                train_batch_noisy[SampleBatch.OBS] = observation_noisy

                # Append the training batches to the set
                train_batches["noisy"] = train_batch_noisy

        # Compute the action_logits for all the training batches at onces
        train_batch_all = {}
        for key in train_batch:
            if key == "infos":
                continue
            train_batch_all[key] = torch.cat([s[key] for s in train_batches.values()], dim=0)
        action_logits_all, state = model(train_batch_all)
        values_all = model.value_function()
        action_mirror_logits_all = model.mirror_function()

        action_logits = dict(zip(train_batches.keys(), torch.chunk(
            action_logits_all, len(train_batches), dim=0)))
        values = dict(zip(train_batches.keys(), torch.chunk(
            values_all, len(train_batches), dim=0)))
        action_mirror_logits = dict(zip(train_batches.keys(), torch.chunk(
            action_mirror_logits_all, len(train_batches), dim=0)))

        action_true_logits = action_logits["true"]
        curr_action_dist = dist_class(action_true_logits, model)

        # RNN case: Mask away 0-padded chunks at end of time axis.
        if state:
            B = len(train_batch[SampleBatch.SEQ_LENS])
            max_seq_len = action_true_logits.shape[0] // B
            mask = sequence_mask(
                train_batch[SampleBatch.SEQ_LENS],
                max_seq_len,
                time_major=model.is_time_major(),
            )
            mask = torch.reshape(mask, [-1])
            num_valid = torch.sum(mask)
            def reduce_mean_valid(t):
                return torch.sum(t[mask]) / num_valid

        # non-RNN case: No masking.
        else:
            mask = None
            reduce_mean_valid = torch.mean

        prev_action_dist = dist_class(
            train_batch[SampleBatch.ACTION_DIST_INPUTS], model
        )

        logp_ratio = torch.exp(
            curr_action_dist.logp(train_batch[SampleBatch.ACTIONS])
            - train_batch[SampleBatch.ACTION_LOGP]
        )

        # Only calculate kl loss if necessary (kl-coeff > 0.0).
        if self.config["kl_coeff"] > 0.0:
            action_kl = prev_action_dist.kl(curr_action_dist)
            mean_kl_loss = reduce_mean_valid(action_kl)
            # TODO smorad: should we do anything besides warn? Could discard KL term
            # for this update
            warn_if_infinite_kl_divergence(self, mean_kl_loss)
        else:
            mean_kl_loss = torch.tensor(0.0, device=logp_ratio.device)

        curr_entropy = curr_action_dist.entropy()
        mean_entropy = reduce_mean_valid(curr_entropy)

        surrogate_loss = torch.min(
            train_batch[Postprocessing.ADVANTAGES] * logp_ratio,
            train_batch[Postprocessing.ADVANTAGES]
            * torch.clamp(
                logp_ratio, 1 - self.config["clip_param"], 1 + self.config["clip_param"]
            ),
        )

        if self.config["enable_symmetry_surrogate_loss"]:

            assert self.action_mirror_mat is not None
            assert isinstance(self.action_space, gym.spaces.Box)

            # Get the mirror policy probability distribution
            # i.e. ( x -> pi( x | mirrored observation ) )
            curr_action_mirror_dist = dist_class(action_mirror_logits["true"], model)

            # The implementation assumes, at any time t, under the policy,
            # the probability to be in state_t is equal to the probability to
            # be in the mirrored state_t. Otherwise, their ratio needs to be
            # added.
            mirror_logp_ratio = torch.exp(
                curr_action_mirror_dist.logp(
                    _compute_mirrored_value(train_batch[SampleBatch.ACTIONS],
                                            self.action_space,
                                            self.action_mirror_mat))
                - train_batch[SampleBatch.ACTION_LOGP]
            )

            symmetry_surrogate_loss = torch.min(
                train_batch[Postprocessing.ADVANTAGES] * mirror_logp_ratio,
                train_batch[Postprocessing.ADVANTAGES]* torch.clamp(
                    mirror_logp_ratio,
                    1 - self.config["clip_param"],
                    1 + self.config["clip_param"]))

            surrogate_loss += symmetry_surrogate_loss
            model.tower_stats["symmetry_surrogate_loss"] =  torch.mean(symmetry_surrogate_loss)

        # Compute a value function loss.
        if self.config["use_critic"]:
            value_fn_out = values["true"]
            vf_loss = torch.pow(
                value_fn_out - train_batch[Postprocessing.VALUE_TARGETS], 2.0
            )
            vf_loss_clipped = torch.clamp(vf_loss, 0, self.config["vf_clip_param"])
            mean_vf_loss = reduce_mean_valid(vf_loss_clipped)
        # Ignore the value function.
        else:
            value_fn_out = torch.tensor(0.0).to(surrogate_loss.device)
            vf_loss_clipped = mean_vf_loss = torch.tensor(0.0).to(surrogate_loss.device)

        total_loss = reduce_mean_valid(
            -surrogate_loss
            + self.config["vf_loss_coeff"] * vf_loss_clipped
            - self.entropy_coeff * curr_entropy
        )

        # Add mean_kl_loss (already processed through `reduce_mean_valid`),
        # if necessary.
        if self.config["kl_coeff"] > 0.0:
            total_loss += self.kl_coeff * mean_kl_loss

        action_true_mean = get_action_mean(
            model, dist_class, action_true_logits)

        # Add symmetric regularisation,
        # if necessary.
        if self.config["symmetric_policy_reg"] > 0.0:
            action_mirror_true_logits = action_mirror_logits["true"]
            action_mirror_mean = get_action_mean(
                model, dist_class, action_mirror_true_logits)
            action_revert_mean = _compute_mirrored_value(
                action_mirror_mean,
                self.action_space,
                self.action_mirror_mat)

            symmetric_policy_reg = reduce_mean_valid(
                (action_revert_mean - action_true_mean) ** 2)
            total_loss += self.config["symmetric_policy_reg"] * symmetric_policy_reg
            model.tower_stats["symmetric_policy_reg"] = symmetric_policy_reg

        # Add CAPS temporal regularisation,
        # if necessary.
        if self.config["caps_temporal_reg"] > 0.0:
            action_dist_prev_logits = train_batch["prev_action_dist_inputs"]
            action_dist_logits = train_batch["action_dist_inputs"]

            action_dist_prev_mean = get_action_mean(
                model, dist_class, action_dist_prev_logits)
            action_dist_mean = get_action_mean(
                model, dist_class, action_dist_logits)

            action_temporal_delta = (action_dist_prev_mean - action_dist_mean).abs()
            caps_temporal_reg = reduce_mean_valid(action_temporal_delta)
            total_loss += \
                self.config["caps_temporal_reg"] * caps_temporal_reg
            model.tower_stats["caps_temporal_reg"] = caps_temporal_reg

        # Add CAPS spatial regularisation,
        # if necessary.
        if self.config["caps_spatial_reg"] > 0.0:
            action_noisy_logits = action_logits["noisy"]
            action_noisy_mean = get_action_mean(
                model, dist_class, action_noisy_logits)

            action_spatial_delta = torch.sum(
                (action_noisy_mean - action_true_mean) ** 2, dim=1)
            caps_spatial_reg = reduce_mean_valid(action_spatial_delta)
            total_loss += \
                self.config["caps_spatial_reg"] * caps_spatial_reg
            model.tower_stats["caps_spatial_reg"] = caps_spatial_reg

        # Add actor l2-regularization loss,
        #if necessary.
        if self.config["l2_reg"] > 0.0:
            # Add actor l2-regularization loss
            l2_reg = 0.0
            assert isinstance(model, torch.nn.Module)
            for name, params in model.named_parameters():
                if not name.endswith("bias") and params.requires_grad:
                    l2_reg += l2_loss(params)

            # Add l2-regularization loss to total loss
            model.tower_stats["l2_reg"] = l2_reg
            total_loss += self.config["l2_reg"] * l2_reg

        # Store values for stats function in model (tower), such that for
        # multi-GPU, we do not override them during the parallel loss phase.
        model.tower_stats["total_loss"] = total_loss
        model.tower_stats["mean_policy_loss"] = reduce_mean_valid(-surrogate_loss)
        model.tower_stats["mean_vf_loss"] = mean_vf_loss
        model.tower_stats["vf_explained_var"] = explained_variance(
            train_batch[Postprocessing.VALUE_TARGETS], value_fn_out
        )
        model.tower_stats["mean_entropy"] = mean_entropy
        model.tower_stats["mean_kl_loss"] = mean_kl_loss

        return total_loss

    @override(_PPOTorchPolicy)
    def stats_fn(self, train_batch: SampleBatch) -> Dict[str, TensorType]:
        """Add regularization statistics.
        """
        stats_dict = super().stats_fn(train_batch)

        if self.config["enable_symmetry_surrogate_loss"]:
            stats_dict["symmetry_surrogate_loss"] = torch.mean(
                torch.stack(self.get_tower_stats("symmetry_surrogate_loss")))
        if self.config["symmetric_policy_reg"] > 0.0:
            stats_dict["symmetry"] = torch.mean(
                torch.stack(self.get_tower_stats("symmetric_policy_reg")))
        if self.config["temporal_barrier_reg"] > 0.0:
            stats_dict["temporal_barrier"] = torch.mean(
                torch.stack(self.get_tower_stats("temporal_barrier_reg")))
        if self.config["caps_temporal_reg"] > 0.0:
            stats_dict["temporal_smoothness"] = torch.mean(
                torch.stack(self.get_tower_stats("caps_temporal_reg")))
        if self.config["caps_spatial_reg"] > 0.0:
            stats_dict["spatial_smoothness"] = torch.mean(
                torch.stack(self.get_tower_stats("caps_spatial_reg")))
        if self.config["caps_global_reg"] > 0.0:
            stats_dict["global_smoothness"] = torch.mean(
                torch.stack(self.get_tower_stats("caps_global_reg")))
        if self.config["l2_reg"] > 0.0:
            stats_dict["l2_reg"] = torch.mean(
                torch.stack(self.get_tower_stats("l2_reg")))

        return convert_to_numpy(stats_dict)


__all__ = [
    "PPOConfig",
    "PPOTorchPolicy",
    "PPO"
]