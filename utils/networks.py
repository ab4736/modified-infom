from typing import Any, Optional, Sequence

import distrax
import flax.struct
import flax.linen as nn
import jax
import jax.numpy as jnp


def default_init(scale=1.0):
    """Default kernel initializer."""
    return nn.initializers.variance_scaling(scale, 'fan_avg', 'uniform')


def ensemblize(cls, num_qs, out_axes=0, **kwargs):
    """Ensemblize a module."""
    return nn.vmap(
        cls,
        variable_axes={'params': 0},
        split_rngs={'params': True},
        in_axes=None,
        out_axes=out_axes,
        axis_size=num_qs,
        **kwargs,
    )


class Identity(nn.Module):
    """Identity layer."""

    def __call__(self, x):
        return x


class MLP(nn.Module):
    """Multi-layer perceptron.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        activations: Activation function.
        activate_final: Whether to apply activation to the final layer.
        kernel_init: Kernel initializer.
        layer_norm: Whether to apply layer normalization.
    """

    hidden_dims: Sequence[int]
    activations: Any = nn.gelu
    activate_final: bool = False
    kernel_init: Any = default_init()
    layer_norm: bool = False

    @nn.compact
    def __call__(self, x):
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=self.kernel_init)(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)
                if self.layer_norm:
                    x = nn.LayerNorm()(x)
            if i == len(self.hidden_dims) - 2:
                self.sow('intermediates', 'feature', x)
        return x


class Param(nn.Module):
    """Scalar parameter module."""

    shape: Sequence[int] = ()
    init_value: float = 0.0

    @nn.compact
    def __call__(self):
        return self.param('value', init_fn=lambda key: jnp.full(self.shape, self.init_value))


class LogParam(nn.Module):
    """Scalar parameter module with log scale."""

    init_value: float = 1.0

    @nn.compact
    def __call__(self):
        log_value = self.param('log_value', init_fn=lambda key: jnp.full((), jnp.log(self.init_value)))
        return jnp.exp(log_value)


@flax.struct.dataclass
class MixtureIntentionDistribution:
    """Lightweight mixture of diagonal Gaussians."""

    means: jnp.ndarray
    log_stds: jnp.ndarray
    logits: jnp.ndarray

    def mixing_probs(self):
        return jax.nn.softmax(self.logits, axis=-1)

    def sample(self, seed, sample_shape=()):
        if sample_shape is None:
            sample_shape = ()
        elif isinstance(sample_shape, int):
            sample_shape = (sample_shape,)
        else:
            sample_shape = tuple(sample_shape)

        eps = jax.random.normal(seed, shape=sample_shape + self.means.shape)
        component_samples = self.means + jnp.exp(self.log_stds) * eps
        weights = self.mixing_probs()
        weights = weights.reshape((1,) * len(sample_shape) + weights.shape + (1,))
        return jnp.sum(weights * component_samples, axis=-2)

    def mean(self):
        weights = self.mixing_probs()[..., None]
        return jnp.sum(weights * self.means, axis=-2)

    def stddev(self):
        weights = self.mixing_probs()[..., None]
        component_vars = jnp.exp(2.0 * self.log_stds)
        second_moment = jnp.sum(weights * (component_vars + self.means**2), axis=-2)
        variance = jnp.maximum(second_moment - self.mean()**2, 1e-8)
        return jnp.sqrt(variance)

    def kl_loss(self):
        kl_per_dim = 0.5 * (
            self.means**2 + jnp.exp(2.0 * self.log_stds) - 1.0 - 2.0 * self.log_stds
        )
        return jnp.sum(kl_per_dim, axis=-1).mean()


class TransformedWithMode(distrax.Transformed):
    """Transformed distribution with mode calculation."""

    def __getattr__(self, name):
        return getattr(self.distribution, name)

    def mode(self):
        return self.bijector.forward(self.distribution.mode())


class Value(nn.Module):
    """Value/critic network.

    This module can be used for both value V(s, g) and critic Q(s, a, g) functions.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        layer_norm: Whether to apply layer normalization.
        num_ensembles: Number of ensemble components.
        encoder: Optional encoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    value_dim: int = 1
    layer_norm: bool = True
    num_ensembles: int = 1
    encoder: nn.Module = None

    def setup(self):
        mlp_class = MLP
        if self.num_ensembles > 1:
            mlp_class = ensemblize(mlp_class, self.num_ensembles)

        value_net = mlp_class(hidden_dims=(*self.hidden_dims, self.value_dim),
                              activate_final=False, layer_norm=self.layer_norm)

        self.value_net = value_net

    def __call__(self, observations, actions=None):
        """Return values or critic values.

        Args:
            observations: Observations.
            actions: Actions (optional).
        """
        if self.encoder is not None:
            inputs = [self.encoder(observations)]
        else:
            inputs = [observations]
        if actions is not None:
            inputs.append(actions)
        inputs = jnp.concatenate(inputs, axis=-1)

        if self.value_dim == 1:
            v = self.value_net(inputs).squeeze(-1)
        else:
            v = self.value_net(inputs)

        return v


class Actor(nn.Module):
    """Gaussian actor network.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        action_dim: Action dimension.
        layer_norm: Whether to apply layer normalization.
        log_std_min: Minimum value of log standard deviation.
        log_std_max: Maximum value of log standard deviation.
        tanh_squash: Whether to squash the action with tanh.
        state_dependent_std: Whether to use state-dependent standard deviation.
        const_std: Whether to use constant standard deviation.
        final_fc_init_scale: Initial scale of the final fully-connected layer.
        encoder: Optional encoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False
    log_std_min: Optional[float] = -5
    log_std_max: Optional[float] = 2
    tanh_squash: bool = False
    state_dependent_std: bool = False
    const_std: bool = True
    final_fc_init_scale: float = 1e-2
    encoder: nn.Module = None

    def setup(self):
        self.actor_net = MLP(self.hidden_dims, activate_final=True, layer_norm=self.layer_norm)
        self.mean_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))
        if self.state_dependent_std:
            self.log_std_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))
        else:
            if not self.const_std:
                self.log_stds = self.param('log_stds', nn.initializers.zeros, (self.action_dim,))

    def __call__(
        self,
        observations,
        latents=None,
        temperature=1.0,
    ):
        """Return action distributions.

        Args:
            observations: Observations.
            latents: Optional intention latent vector to concatenate with observations.
            temperature: Scaling factor for the standard deviation.
        """
        if self.encoder is not None:
            inputs = self.encoder(observations)
        else:
            inputs = observations
        if latents is not None:
            inputs = jnp.concatenate([inputs, latents], axis=-1)
        outputs = self.actor_net(inputs)

        means = self.mean_net(outputs)
        if self.state_dependent_std:
            log_stds = self.log_std_net(outputs)
        else:
            if self.const_std:
                log_stds = jnp.zeros_like(means)
            else:
                log_stds = self.log_stds

        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        distribution = distrax.MultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds) * temperature)
        if self.tanh_squash:
            distribution = TransformedWithMode(distribution, distrax.Block(distrax.Tanh(), ndims=1))

        return distribution


class IntentionEncoder(nn.Module):
    """Transition encoder network.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        latent_dim: Latent dimension.
        layer_norm: Whether to apply layer normalization.
        encoder: Optional encoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    latent_dim: int
    num_components: int = 1
    layer_norm: bool = False
    encoder: nn.Module = None

    def setup(self):
        self.trunk_net = MLP(self.hidden_dims, activate_final=True, layer_norm=self.layer_norm)
        self.mean_net = nn.Dense(self.num_components * self.latent_dim, kernel_init=default_init())
        self.log_std_net = nn.Dense(self.num_components * self.latent_dim, kernel_init=default_init())
        self.logits_net = nn.Dense(self.num_components, kernel_init=default_init())

    def __call__(
        self,
        observations,
        actions,
    ):
        """Return latent distribution.

        Args:
            observations: Observations.
            actions: Actions.
        """
        if self.encoder is not None:
            observations = self.encoder(observations)
        inputs = jnp.concatenate([observations, actions], axis=-1)
        outputs = self.trunk_net(inputs)

        means = self.mean_net(outputs)
        log_stds = self.log_std_net(outputs)
        logits = self.logits_net(outputs)

        batch_shape = means.shape[:-1]
        means = means.reshape(*batch_shape, self.num_components, self.latent_dim)
        log_stds = log_stds.reshape(*batch_shape, self.num_components, self.latent_dim)

        distribution = MixtureIntentionDistribution(means=means, log_stds=log_stds, logits=logits)

        return distribution


class AttentionIntentionEncoder(nn.Module):
    """Transformer-based intention encoder over a window of consecutive transitions.

    Encodes a sequence of K (observation, action) pairs into a latent intention z,
    allowing the model to infer which phase of a multi-step task the agent is in.

    Attributes:
        hidden_dims: Hidden layer dimensions; last dim is the Transformer model dim.
        latent_dim: Dimension of the output latent variable z.
        window_size: Number of consecutive transitions in the context window.
        num_heads: Number of self-attention heads.
        num_layers: Number of Transformer encoder layers.
        layer_norm: Whether to apply layer norm to the input projection.
        encoder: Optional observation encoder (e.g. for pixel inputs).
    """

    hidden_dims: Sequence[int]
    latent_dim: int
    window_size: int = 4
    num_heads: int = 4
    num_layers: int = 2
    layer_norm: bool = False
    encoder: nn.Module = None

    @nn.compact
    def __call__(self, observations, actions):
        # observations: [B, K, obs_dim] or [B, K, H, W, C], actions: [B, K, act_dim]
        if self.encoder is not None:
            B, K = observations.shape[0], observations.shape[1]
            obs_shape = observations.shape[2:]  # (obs_dim,) or (H, W, C)
            observations = self.encoder(observations.reshape(B * K, *obs_shape)).reshape(B, K, -1)

        x = jnp.concatenate([observations, actions], axis=-1)  # [B, K, obs+act]
        model_dim = self.hidden_dims[-1]

        x = nn.Dense(model_dim, kernel_init=default_init())(x)  # [B, K, model_dim]
        if self.layer_norm:
            x = nn.LayerNorm()(x)

        pos_emb = self.param(
            'pos_emb', nn.initializers.normal(0.02), (self.window_size, model_dim))
        x = x + pos_emb  # broadcast over batch

        for _ in range(self.num_layers):
            # Self-attention sub-layer with pre-norm and residual
            h = nn.LayerNorm()(x)
            h = nn.MultiHeadDotProductAttention(
                num_heads=self.num_heads,
                kernel_init=default_init(),
            )(h)
            x = x + h
            # Feed-forward sub-layer with pre-norm and residual
            h = nn.LayerNorm()(x)
            h = nn.Dense(model_dim * 4, kernel_init=default_init())(h)
            h = nn.gelu(h)
            h = nn.Dense(model_dim, kernel_init=default_init())(h)
            x = x + h

        # Pool from the last (most recent) token — encodes the current phase
        h = x[:, -1, :]  # [B, model_dim]

        means = nn.Dense(self.latent_dim, kernel_init=default_init())(h)
        log_stds = nn.Dense(self.latent_dim, kernel_init=default_init())(h)
        log_stds = jnp.clip(log_stds, -5, 2)

        # Return single-component MixtureIntentionDistribution for unified interface
        means = means[:, None, :]       # [B, 1, latent_dim]
        log_stds = log_stds[:, None, :] # [B, 1, latent_dim]
        logits = jnp.zeros((*h.shape[:-1], 1))  # [B, 1]
        return MixtureIntentionDistribution(means=means, log_stds=log_stds, logits=logits)


class BoundaryAttentionIntentionEncoder(nn.Module):
    """Transformer intention encoder with phase-boundary masking.

    Same as AttentionIntentionEncoder but restricts self-attention to within
    detected phase boundaries. Boundaries are identified by large observation
    deltas (adaptive 75th-percentile threshold per window), so the last token
    only attends to transitions in the same behavioral phase.
    """

    hidden_dims: Sequence[int]
    latent_dim: int
    window_size: int = 8
    num_heads: int = 4
    num_layers: int = 2
    layer_norm: bool = False
    encoder: nn.Module = None

    @nn.compact
    def __call__(self, observations, actions):
        # observations: [B, K, obs_dim] or [B, K, H, W, C], actions: [B, K, act_dim]
        if self.encoder is not None:
            B, K = observations.shape[0], observations.shape[1]
            obs_shape = observations.shape[2:]  # (obs_dim,) or (H, W, C)
            observations = self.encoder(observations.reshape(B * K, *obs_shape)).reshape(B, K, -1)

        # Detect phase boundaries via observation deltas
        deltas = jnp.linalg.norm(
            observations[:, 1:, :] - observations[:, :-1, :], axis=-1
        )  # [B, K-1]
        # Adaptive threshold: 75th percentile within each window
        threshold = jnp.percentile(deltas, 75, axis=-1, keepdims=True)  # [B, 1]
        is_boundary = deltas > threshold  # [B, K-1]

        # Assign segment IDs: increment at each boundary
        boundary_padded = jnp.concatenate(
            [jnp.zeros((observations.shape[0], 1), dtype=jnp.int32),
             is_boundary.astype(jnp.int32)], axis=1
        )  # [B, K]
        segment_ids = jnp.cumsum(boundary_padded, axis=1)  # [B, K]

        # Boolean attention mask: attend only within the same segment
        # [B, K, K] -> [B, 1, K, K] to broadcast over heads
        attn_mask = (segment_ids[:, :, None] == segment_ids[:, None, :])  # [B, K, K]
        attn_mask = attn_mask[:, None, :, :]  # [B, 1, K, K]

        x = jnp.concatenate([observations, actions], axis=-1)
        model_dim = self.hidden_dims[-1]

        x = nn.Dense(model_dim, kernel_init=default_init())(x)
        if self.layer_norm:
            x = nn.LayerNorm()(x)

        pos_emb = self.param(
            'pos_emb', nn.initializers.normal(0.02), (self.window_size, model_dim))
        x = x + pos_emb

        for _ in range(self.num_layers):
            h = nn.LayerNorm()(x)
            h = nn.MultiHeadDotProductAttention(
                num_heads=self.num_heads,
                kernel_init=default_init(),
            )(h, mask=attn_mask)
            x = x + h
            h = nn.LayerNorm()(x)
            h = nn.Dense(model_dim * 4, kernel_init=default_init())(h)
            h = nn.gelu(h)
            h = nn.Dense(model_dim, kernel_init=default_init())(h)
            x = x + h

        h = x[:, -1, :]  # last token = most recent transition

        means = nn.Dense(self.latent_dim, kernel_init=default_init())(h)
        log_stds = nn.Dense(self.latent_dim, kernel_init=default_init())(h)
        log_stds = jnp.clip(log_stds, -5, 2)

        means = means[:, None, :]
        log_stds = log_stds[:, None, :]
        logits = jnp.zeros((*h.shape[:-1], 1))
        return MixtureIntentionDistribution(means=means, log_stds=log_stds, logits=logits)


class VectorField(nn.Module):
    """Flow-matching vector field function.

    This module can be used for both value velocity field u(s, g) and critic velocity filed u(s, a, g) functions.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        layer_norm: Whether to apply layer normalization.
        num_ensembles: Number of ensemble components.
        encoder: state encoder.
    """

    vector_dim: int
    hidden_dims: Sequence[int]
    layer_norm: bool = True
    num_ensembles: int = 1
    encoder: nn.Module = None

    def setup(self):
        mlp_class = MLP

        if self.num_ensembles > 1:
            mlp_class = ensemblize(mlp_class, self.num_ensembles)

        self.vector_field_net = mlp_class(
            hidden_dims=(*self.hidden_dims, self.vector_dim),
            activate_final=False, layer_norm=self.layer_norm
        )

    def __call__(self, noisy_goals, times, observations=None, actions=None, latents=None):
        """Return the value/critic velocity field.

        Args:
            noisy_goals: Noisy goals.
            times: Times.
            observations: Observations (Optional).
            actions: Actions (Optional).
        """
        if self.encoder is not None:
            noisy_goals = self.encoder(noisy_goals)

        if self.encoder is not None and observations is not None:
            observations = self.encoder(observations)

        times = times[..., None]
        inputs = [noisy_goals, times]
        if observations is not None:
            inputs.append(observations)
        if actions is not None:
            inputs.append(actions)
        if latents is not None:
            inputs.append(latents)
        inputs = jnp.concatenate(inputs, axis=-1)

        vf = self.vector_field_net(inputs)

        return vf


class GCActor(nn.Module):
    """Goal-conditioned actor.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        action_dim: Action dimension.
        log_std_min: Minimum value of log standard deviation.
        log_std_max: Maximum value of log standard deviation.
        tanh_squash: Whether to squash the action with tanh.
        state_dependent_std: Whether to use state-dependent standard deviation.
        const_std: Whether to use constant standard deviation.
        final_fc_init_scale: Initial scale of the final fully-connected layer.
        gc_encoder: Optional GCEncoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False
    log_std_min: Optional[float] = -5
    log_std_max: Optional[float] = 2
    tanh_squash: bool = False
    state_dependent_std: bool = False
    const_std: bool = True
    final_fc_init_scale: float = 1e-2
    gc_encoder: nn.Module = None

    def setup(self):
        self.actor_net = MLP(
            self.hidden_dims,
            activate_final=True,
            layer_norm=self.layer_norm
        )

        self.mean_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))
        if self.state_dependent_std:
            self.log_std_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))
        else:
            if not self.const_std:
                self.log_stds = self.param('log_stds', nn.initializers.zeros, (self.action_dim,))

    def __call__(
        self,
        observations,
        goals=None,
        goal_encoded=False,
        temperature=1.0,
    ):
        """Return the action distribution.

        Args:
            observations: Observations.
            goals: Goals (optional).
            goal_encoded: Whether the goals are already encoded (optional).
            temperature: Scaling factor for the standard deviation (optional).
        """
        if self.gc_encoder is not None:
            inputs = self.gc_encoder(observations, goals, goal_encoded=goal_encoded)
        else:
            inputs = [observations]
            if goals is not None:
                inputs.append(goals)
            inputs = jnp.concatenate(inputs, axis=-1)
        outputs = self.actor_net(inputs)

        means = self.mean_net(outputs)
        if self.state_dependent_std:
            log_stds = self.log_std_net(outputs)
        else:
            if self.const_std:
                log_stds = jnp.zeros_like(means)
            else:
                log_stds = self.log_stds

        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        distribution = distrax.MultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds) * temperature)
        if self.tanh_squash:
            distribution = TransformedWithMode(distribution, distrax.Block(distrax.Tanh(), ndims=1))

        return distribution


class GCValue(nn.Module):
    """Goal-conditioned value/critic function.

    This module can be used for both value V(s, g) and critic Q(s, a, g) functions.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        value_dim: Value dimension.
        layer_norm: Whether to apply layer normalization.
        num_ensembles: Number of ensemble components.
        gc_encoder: Optional GCEncoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    value_dim: int = 1
    num_residual_blocks: int = 1
    layer_norm: bool = True
    num_ensembles: int = 1
    gc_encoder: nn.Module = None

    def setup(self):
        mlp_class = MLP

        if self.num_ensembles > 1:
            mlp_class = ensemblize(mlp_class,  self.num_ensembles)

        self.value_net = mlp_class(
            (*self.hidden_dims, self.value_dim),
            activate_final=False,
            layer_norm=self.layer_norm
        )

    def __call__(self, observations, goals=None, actions=None, goal_encoded=False):
        """Return the value/critic function.

        Args:
            observations: Observations.
            goals: Goals (optional).
            actions: Actions (optional).
            goal_encoded: Whether the goals are already encoded (optional).
        """
        if self.gc_encoder is not None:
            inputs = [self.gc_encoder(observations, goals, goal_encoded=goal_encoded)]
        else:
            inputs = [observations]
            if goals is not None:
                inputs.append(goals)
        if actions is not None:
            inputs.append(actions)
        inputs = jnp.concatenate(inputs, axis=-1)

        if self.value_dim == 1:
            v = self.value_net(inputs).squeeze(-1)
        else:
            v = self.value_net(inputs)

        return v


class GCBilinearValue(nn.Module):
    """Goal-conditioned bilinear value/critic function.

    This module computes the value function as V(s, g) = phi(s)^T psi(g) / sqrt(d) or the critic function as
    Q(s, a, g) = phi(s, a)^T psi(g) / sqrt(d), where phi and psi output d-dimensional vectors.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        latent_dim: Latent dimension.
        layer_norm: Whether to apply layer normalization.
        num_ensembles: Number of ensemble components.
        value_exp: Whether to exponentiate the value. Useful for contrastive learning.
        state_encoder: Optional state encoder.
        goal_encoder: Optional goal encoder.
    """

    hidden_dims: Sequence[int]
    latent_dim: int
    layer_norm: bool = True
    num_ensembles: int = 1
    value_exp: bool = False
    state_encoder: nn.Module = None
    goal_encoder: nn.Module = None

    def setup(self) -> None:
        mlp_class = MLP

        if self.num_ensembles > 1:
            mlp_class = ensemblize(mlp_class,  self.num_ensembles)

        self.phi = mlp_class(
            (*self.hidden_dims, self.latent_dim),
            activate_final=False, layer_norm=self.layer_norm
        )
        self.psi = mlp_class(
            (*self.hidden_dims, self.latent_dim),
            activate_final=False, layer_norm=self.layer_norm
        )

    def __call__(self, observations, goals, actions=None, info=False):
        """Return the value/critic function.

        Args:
            observations: Observations.
            goals: Goals.
            actions: Actions (optional).
            info: Whether to additionally return the representations phi and psi.
        """
        if self.state_encoder is not None:
            observations = self.state_encoder(observations)
        if self.goal_encoder is not None:
            goals = self.goal_encoder(goals)

        if actions is None:
            phi_inputs = observations
        else:
            phi_inputs = jnp.concatenate([observations, actions], axis=-1)

        phi = self.phi(phi_inputs)
        psi = self.psi(goals)

        if len(phi.shape) == 2:  # Non-ensemble.
            v = jnp.einsum('ik,jk->ij', phi, psi) / jnp.sqrt(self.latent_dim)
        else:
            v = jnp.einsum('eik,ejk->eij', phi, psi) / jnp.sqrt(self.latent_dim)

        if self.value_exp:
            v = jnp.exp(v)

        if info:
            return v, phi, psi
        else:
            return v


class GCMetricValue(nn.Module):
    """Metric value function.

    This module computes the value function as || \phi(s) - \phi(g) ||_2.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        latent_dim: Latent dimension.
        layer_norm: Whether to apply layer normalization.
        encoder: Optional state/goal encoder.
    """

    hidden_dims: Sequence[int]
    latent_dim: int
    layer_norm: bool = True
    num_ensembles: int = 1
    encoder: nn.Module = None

    def setup(self) -> None:
        network_module = MLP
        if self.num_ensembles > 1:
            network_module = ensemblize(network_module,  self.num_ensembles)
        self.phi = network_module((*self.hidden_dims, self.latent_dim), activate_final=False, layer_norm=self.layer_norm)

    def __call__(self, observations, goals, is_phi=False, info=False):
        """Return the metric value function.

        Args:
            observations: Observations.
            goals: Goals.
            is_phi: Whether the inputs are already encoded by phi.
            info: Whether to additionally return the representations phi_s and phi_g.
        """
        if is_phi:
            phi_s = observations
            phi_g = goals
        else:
            if self.encoder is not None:
                observations = self.encoder(observations)
                goals = self.encoder(goals)
            phi_s = self.phi(observations)
            phi_g = self.phi(goals)

        squared_dist = ((phi_s - phi_g) ** 2).sum(axis=-1)
        v = -jnp.sqrt(jnp.maximum(squared_dist, 1e-12))

        if info:
            return v, phi_s, phi_g
        else:
            return v
