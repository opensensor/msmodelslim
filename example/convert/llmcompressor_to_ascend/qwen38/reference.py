# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Qwen Team and The HuggingFace Inc. team.
# Adapted from huggingface/transformers bd15bc95a89e728bbc1224084eb3b5829428c353.
# See UPSTREAM.md for scope and modifications.
"""Upstream eager text primitives; no remote kernels or serving integration."""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, is_torchdynamo_exporting

from .configuration import Qwen4ExpTextConfig

maybe_autocast = torch.autocast


class Qwen4ExpTextRotaryEmbedding(nn.Module):
    def __init__(self, config: Qwen4ExpTextConfig, device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config

        self.rope_type = self.config.rope_parameters["rope_type"]
        rope_init_fn: Callable = self.compute_default_rope_parameters
        if self.rope_type != "default":
            rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = rope_init_fn(self.config, device)

        self.inv_freq = nn.Buffer(inv_freq, persistent=False)
        self.original_inv_freq = nn.Buffer(inv_freq.clone(), persistent=False)
        self.mrope_section = config.rope_parameters.get("mrope_section", [11, 11, 10])

    @staticmethod
    def compute_default_rope_parameters(
        config: Qwen4ExpTextConfig, device=None, **kwargs
    ) -> tuple[torch.Tensor, float]:
        """
        Computes the inverse frequencies according to the original RoPE implementation
        Args:
            config ([`~transformers.PreTrainedConfig`]):
                The model configuration.
        Returns:
            Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
            post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
        """
        base = config.rope_parameters["rope_theta"]
        partial_rotary_factor = config.rope_parameters.get("partial_rotary_factor", 1.0)
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        dim = int(head_dim * partial_rotary_factor)

        attention_factor = 1.0  # Unused in this type of RoPE
        # Compute the inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        return inv_freq.to(device), attention_factor

    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):
        # In contrast to other models, Qwen4ExpText has different position ids for the grids
        # So we expand the inv_freq to shape (3, ...)
        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        position_ids_expanded = position_ids[:, :, None, :].float()  # shape (3, bs, 1, positions)

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with maybe_autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
            cos = freqs.cos() * self.attention_scaling
            sin = freqs.sin() * self.attention_scaling

        sin = self.recomposition_frequencies(sin)
        cos = self.recomposition_frequencies(cos)
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    def recomposition_frequencies(self, freq):
        """
        Recompose the frequencies into the final spatial layout used per each grid.
        """
        freqs_thw = freq[0]  # just overwrite the first dimension T
        for dim, offset in enumerate((1, 2), start=1):  # H, W
            length = self.mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_thw[..., idx] = freq[dim, ..., idx]
        return torch.cat((freqs_thw, freqs_thw), dim=-1)


class Qwen4ExpTextRMSNorm(nn.Module):
    def __init__(self, dim: int, group_size: int | None = None, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))
        self.group_size = group_size
        if group_size is not None and dim % group_size != 0:
            raise ValueError(f"hidden_size ({dim}) must be divisible by group_size ({group_size}).")

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        if self.group_size is not None:
            x = x.reshape(*x.shape[:-1], -1, self.group_size)
        out = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return out.flatten(-2) if self.group_size is not None else out

    def forward(self, x):
        output = self._norm(x.float())
        # Llama does x.to(float16) * w whilst Qwen4ExpText is (x * w).to(float16)
        # See https://github.com/huggingface/transformers/pull/29402
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


# NOTE: the FLA package does not re-cast to `input_dtype` in its implementation, maybe we should do the same
class Qwen4ExpTextRMSNormGated(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6, activation: str = "silu") -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.activation = activation

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        # Norm before gate
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states.to(input_dtype)
        hidden_states = hidden_states * ACT2FN[self.activation](gate.to(torch.float32))

        return hidden_states.to(input_dtype)


def apply_mask_to_padding_states(hidden_states, attention_mask):
    """
    Tunes out the hidden states for padding tokens, see https://github.com/state-spaces/mamba/issues/66
    """
    # NOTE: attention mask is a 2D boolean tensor
    if attention_mask is not None:
        dtype = hidden_states.dtype
        hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)

    return hidden_states


def causal_conv1d_update(
    hidden_states: torch.Tensor,
    conv_state: torch.Tensor,
    weight: nn.Parameter,
    bias: nn.Parameter | None = None,
    activation: str | None = None,
):
    _, hidden_size, seq_len = hidden_states.shape
    state_len = conv_state.shape[-1]

    hidden_states_new = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
    conv_state.copy_(hidden_states_new[:, :, -state_len:])
    out = F.conv1d(hidden_states_new, weight.unsqueeze(1), bias, padding=0, groups=hidden_size)
    out = out[:, :, -seq_len:]
    if activation is not None:
        out = ACT2FN[activation](out)
    return out.to(hidden_states.dtype)


def causal_conv1d_fn(
    hidden_states: torch.Tensor,
    weight: nn.Parameter,
    bias: nn.Parameter | None = None,
    activation: str | None = None,
    **kwargs,
):
    _, hidden_size, seq_len = hidden_states.shape
    padding = weight.shape[-1] - 1

    out = F.conv1d(
        hidden_states.to(weight.dtype),
        weight=weight.unsqueeze(1),
        bias=bias,
        padding=padding,
        groups=hidden_size,
    )[:, :, :seq_len]
    if activation is not None:
        out = ACT2FN[activation](out)
    return out.to(hidden_states.dtype)


# NOTE: the FLA package computes `x / torch.sqrt((x * x).sum(dim=dim, keepdim=True) + eps)` instead, so if we align
# with the GatedRMSNorm, maybe we can make that change as well.
def l2norm(x: torch.FloatTensor, dim: int = -1, eps: float = 1e-6):
    """This function is intended to align with the l2norm implementation in the FLA library."""
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def torch_chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Computes the gated delta rule, by chunking along the sequence dimension.
    Args:
        query: Query tensor of shape [batch_size, sequence_length, num_k_heads, k_head_dim]
        key: Key tensor of shape [batch_size, sequence_length, num_k_heads, k_head_dim]
        value: Value tensor of shape [batch_size, sequence_length, num_v_heads, v_head_dim]. num_v_heads can be equal
            to num_k_heads, same for v_head_dim and k_head_dim.
        g: Decay (in log space) tensor of shape [batch_size, sequence_length, num_v_heads]: the recurrent state is
            multiplied by exp(g) at each step, so entries must be <= 0.
        beta: Beta tensor of shape [batch_size, sequence_length, num_v_heads]
        chunk_size: Size of the chunks along the sequence dimension.
        initial_state: The recurrent state, an optional tensor of shape [batch_size, num_v_heads, k_head_dim, v_head_dim]
        output_final_state: Whether to output the new recurrent state along with the output.
        use_qk_l2norm_in_kernel: If this flag is set to True, query and key vectors are L2-normalized.
    Returns:
        - The output tensor of shape [batch_size, sequence_length, num_v_heads, v_head_dim]
        - Either None or the new recurrent state tensor of shape [batch_size, num_v_heads, k_head_dim, v_head_dim]
    """
    initial_dtype = query.dtype
    batch_size, sequence_length, _, k_head_dim = key.shape
    num_v_heads, v_head_dim = value.shape[-2:]
    recurrent_state_shape = (batch_size, num_v_heads, k_head_dim, v_head_dim)
    padded_output_shape = (batch_size, num_v_heads, -1, v_head_dim)  # -1 is the padded sequence length
    decay = g  # rename for clarity: argument name must stay "g" to match flash_linear_attention's API

    # Make sure all tensors are fp32 and reshape them to [batch_size, num_*_heads, seqlen, ...]
    query, key, value, beta, decay = [
        x.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format)
        for x in (query, key, value, beta, decay)
    ]
    # If enabled, normalize query and key vectors (in fp32 to match the FLA library)
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    # And always normalize queries by the head dimension
    scaling = query.shape[-1] ** -0.5
    query = query * scaling

    # Pad sequence length to be a multiple of chunk_size. Padding is described as (left_pad, right_pad) for each dim.
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query, key, value = (F.pad(x, (0, 0, 0, pad_size)) for x in (query, key, value))
    beta, decay = (F.pad(x, (0, pad_size)) for x in (beta, decay))

    total_sequence_length = sequence_length + pad_size
    num_chunks = total_sequence_length // chunk_size

    # Apply beta to K and V, which is the "learning rate" of the recurrent state for a given token, ie. how much the new
    # state influence the old state. Beta is often normalized to (0, 1) where 0 = no update; 1 = overwrite old state.
    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)

    # Reshape all tensors to chunk the sequence dimension (adds a new dimension of size chunk_size)
    query, key, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, k_beta, v_beta)
    ]
    decay = decay.reshape(decay.shape[0], decay.shape[1], -1, chunk_size)

    # Create a chunk-sized strictly upper triangular mask, ie. the mask of what a causal chunk may not attend to
    strictly_upper_mask = torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device).triu(1)

    # Cumulative decay within each chunk (dim 3 is the position inside the chunk). Since decay is in log space,
    # cum_decay[..., t] is the log of a product of decays between the start of the chunk and position t
    cum_decay = decay.cumsum(dim=3)

    # First phase: compute intra-chunk quantities.
    # The pairwise decays: pairwise_decay[..., i, j] = exp(cum_decay_i - cum_decay_j) is the decay accumulated between
    # positions j and i of a chunk. Positive values are masked to -inf before exp to avoid overflow
    pairwise_decay = cum_decay.unsqueeze(4) - cum_decay.unsqueeze(3)
    pairwise_decay = pairwise_decay.masked_fill(strictly_upper_mask, float("-inf"))
    pairwise_decay = pairwise_decay.exp()  # with the exp, we exit log space, so we can apply this decay to the states

    # Compute auxiliary tensors: the Upper Triangular (ut) transform system and the intra-chunk attn (QK dot product)
    ut_system = (k_beta @ key.transpose(-1, -2)) * pairwise_decay
    intra_chunk_attn = (query @ key.transpose(-1, -2)) * pairwise_decay
    decayed_k_beta = k_beta * cum_decay.exp().unsqueeze(-1)

    # Gated delta attention uses a UT transform to condense several delta rule updates into a few matmuls. After the UT
    # system is solved, we can then compute the new_values (called "u" in the DeltaNet paper) and the decayed keys
    # reading the old state (k_cumdecay). In the update, the part of new_values that the old state already predicts is
    # subtracted out, so that only the correction is written to the recurrent state: this is the delta rule.

    # Not all export targets support the fast triangular solver, so we build the inverse by forward substitution then
    if not is_torchdynamo_exporting():
        new_values = torch.linalg.solve_triangular(ut_system, v_beta, upper=False, unitriangular=True)
        k_cumdecay = torch.linalg.solve_triangular(ut_system, decayed_k_beta, upper=False, unitriangular=True)
    else:
        ut_system = -ut_system.tril(-1)  # ut_system is masked to only keep the strictly lower triangle
        for i in range(1, chunk_size):
            row = ut_system[..., i, :i].clone()
            sub = ut_system[..., :i, :i].clone()
            ut_system[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
        ut_system = ut_system + torch.eye(chunk_size, dtype=ut_system.dtype, device=ut_system.device)
        new_values, k_cumdecay = ut_system @ v_beta, ut_system @ decayed_k_beta

    if initial_state is None:
        last_recurrent_state = torch.zeros(recurrent_state_shape, dtype=new_values.dtype, device=new_values.device)
    else:
        last_recurrent_state = initial_state.to(new_values)
    core_attn_out = torch.zeros_like(new_values)

    # Apply decay once rather than in each chunk
    query = query * cum_decay.exp().unsqueeze(-1)
    key = key * (cum_decay[..., -1:] - cum_decay).exp().unsqueeze(-1)
    chunk_decay = cum_decay[..., -1].exp()[..., None, None]

    # Second phase: the sequential scan over chunks
    for i in range(num_chunks):
        # Compute attention output for the current chunk: add the read of the previous recurrent state
        # (inter_chunk_attn) with the within-chunk attention (intra_chunk_attn)
        v_new = new_values[:, :, i] - k_cumdecay[:, :, i] @ last_recurrent_state
        inter_chunk_attn = query[:, :, i] @ last_recurrent_state
        core_attn_out[:, :, i] = inter_chunk_attn + intra_chunk_attn[:, :, i] @ v_new
        # Update the recurrent state: new recurrent state (S_t+1) = decayed old state (S_t * (I-βkk^T)) + update (βvk^T)
        last_recurrent_state = last_recurrent_state * chunk_decay[:, :, i] + key[:, :, i].transpose(-1, -2) @ v_new

    # Discard the final state if not requested
    last_recurrent_state = None if not output_final_state else last_recurrent_state
    # Reshape the output to the orignal shape: flatten the chunk dimension, then drop padding
    core_attn_out = core_attn_out.reshape(padded_output_shape)
    core_attn_out = core_attn_out[:, :, :sequence_length]
    # Convert back to the original shape [batch_size, sequence_length, num_v_heads, v_head_dim] and dtype
    core_attn_out = core_attn_out.transpose(1, 2).to(initial_dtype, memory_format=torch.contiguous_format)
    return core_attn_out, last_recurrent_state


def torch_recurrent_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Computes linear attention using the gated delta rule, by iterating over each token in the sequence dimension.
    Same args and return value as torch_chunk_gated_delta_rule, except for `chunk_size` because the sequence dim is not
    chunked.
    """
    initial_dtype = query.dtype
    batch_size, sequence_length, _, k_head_dim = key.shape
    num_v_heads, v_head_dim = value.shape[-2:]
    decay = g  # rename for clarity: argument name must stay "g" to match flash_linear_attention's API

    # Make sure all tensors are fp32 and reshape them to [batch_size, num_*_heads, seqlen, ...]
    query, key, value, beta, decay = [
        x.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format)
        for x in (query, key, value, beta, decay)
    ]
    # If enabled, normalize query and key vectors (done once in fp32 for better accuracy)
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)

    # And always normalize queries by the head dimension
    query = query / (query.shape[-1] ** 0.5)

    # Create the storage for the last recurrent state, which will be updated in place. If a previous state is provided,
    # it is the starting point, otherwise start with a zeroed buffer.
    if initial_state is None:
        recurrent_state_shape = (batch_size, num_v_heads, k_head_dim, v_head_dim)
        last_recurrent_state = torch.zeros(recurrent_state_shape, dtype=value.dtype, device=value.device)
    else:
        last_recurrent_state = initial_state.to(value)
    core_attn_out = torch.zeros_like(value)

    # Loop over each token and update the recurrent state
    for i in range(sequence_length):
        q_t, k_t, v_t = query[:, :, i], key[:, :, i], value[:, :, i]
        # Decay the recurrent state
        decay_t = decay[:, :, i].exp()[..., None, None]
        last_recurrent_state = last_recurrent_state * decay_t
        # Update the recurrent state with the current token
        beta_t = beta[:, :, i].unsqueeze(-1)
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        # And use it to compute the attention output for the current token
        core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)

    # Discard the final state if not requested
    last_recurrent_state = None if not output_final_state else last_recurrent_state
    # Convert back to the original shape [batch_size, sequence_length, num_v_heads, v_head_dim] and dtype
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


class Qwen4ExpTextGatedDeltaNet(nn.Module):
    def __init__(self, config: Qwen4ExpTextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads

        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_idx = layer_idx
        self.activation = config.hidden_act
        self.layer_norm_epsilon = config.rms_norm_eps

        # QKV
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
        )

        # time step projection (discretization)
        # instantiate once and copy inv_dt in init_weights of PretrainedModel
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))

        # Lower bound kept away from 0 so log(A) never becomes -inf
        A = torch.empty(self.num_v_heads).uniform_(0.01, 16)
        self.A_log = nn.Parameter(torch.log(A))
        self.norm = Qwen4ExpTextRMSNormGated(
            self.head_v_dim, eps=self.layer_norm_epsilon, activation=config.output_gate_type or config.hidden_act
        )
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

        self.layer_type = config.layer_types[layer_idx]

        self.in_proj_qkv = nn.Linear(self.hidden_size, self.key_dim * 2 + self.value_dim, bias=False)
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: Cache | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)

        # Set up dimensions for reshapes later
        batch_size, seq_len, _ = hidden_states.shape
        use_precomputed_states = cache_params is not None and cache_params.has_previous_state(
            self.layer_idx, state_idx=0
        )

        mixed_qkv = self.in_proj_qkv(hidden_states)
        mixed_qkv = mixed_qkv.transpose(1, 2)

        z = self.in_proj_z(hidden_states)
        z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)

        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        if use_precomputed_states and seq_len == 1 and not cache_params.layers[self.layer_idx].record_past:
            conv_state = cache_params.layers[self.layer_idx].conv_states[0]
            # Single-token cached decode: the fused per-step kernel updates the conv state in-place.
            mixed_qkv = causal_conv1d_update(
                mixed_qkv,
                conv_state,
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
                self.activation,
            )
        else:
            if cache_params is not None:
                mixed_qkv = cache_params.update_conv_state(
                    mixed_qkv, self.layer_idx, conv_kernel_size=self.conv_kernel_size
                )

            mixed_qkv = causal_conv1d_fn(
                mixed_qkv,
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
                activation=self.activation,
                **kwargs,
            )

            # Drop the additional previous states
            if cache_params is not None:
                mixed_qkv = mixed_qkv[:, :, -seq_len:]

        mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = torch.split(
            mixed_qkv,
            [
                self.key_dim,
                self.key_dim,
                self.value_dim,
            ],
            dim=-1,
        )

        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

        beta = b.sigmoid()
        # If the model is loaded in fp16, without the .float() here, A might be -inf
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        recurrent_state = cache_params.layers[self.layer_idx].recurrent_states[0] if use_precomputed_states else None
        if use_precomputed_states and seq_len == 1:
            core_attn_out, last_recurrent_state = torch_recurrent_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=kwargs.pop("cu_seq_lens_q", None),
                **kwargs,
            )
        else:
            core_attn_out, last_recurrent_state = torch_chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=kwargs.pop("cu_seq_lens_q", None),
                **kwargs,
            )

        # Update cache
        if cache_params is not None:
            cache_params.update_recurrent_state(last_recurrent_state, self.layer_idx)

        # reshape input data into 2D tensor
        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)

        output = self.out_proj(core_attn_out)
        return output


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k=None, cos=None, sin=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors, or only the queries if the keys are not provided.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor if provided.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    rotary_dim = cos.shape[-1]

    # Keep half or full tensor for later concatenation
    q_rope, q_nope = q[..., :rotary_dim], q[..., rotary_dim:]
    # Apply rotary embeddings on the first half or full tensor
    q_rope = (q_rope * cos) + (rotate_half(q_rope) * sin)
    # Concatenate back to full shape
    q_rotated = torch.cat([q_rope, q_nope], dim=-1)

    if k is not None:
        k_rope, k_nope = k[..., :rotary_dim], k[..., rotary_dim:]
        k_rope = (k_rope * cos) + (rotate_half(k_rope) * sin)
        k_rotated = torch.cat([k_rope, k_nope], dim=-1)
        return q_rotated, k_rotated
    else:
        return q_rotated


class Qwen4ExpTextQSAIndexer(nn.Module):
    """Select QSA token indices from compressed key blocks."""

    def __init__(self, config: Qwen4ExpTextConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.index_n_heads = config.indexer_n_heads
        self.index_kv_heads = config.indexer_kv_heads
        self.index_head_dim = config.indexer_head_dim
        self.token_budget = config.indexer_budget
        self.compress_ratio = config.indexer_compress_ratio
        self.block_topk = self.token_budget // self.compress_ratio
        self.index_qk_proj = nn.Linear(
            config.hidden_size,
            (self.index_n_heads + self.index_kv_heads) * self.index_head_dim,
            bias=False,
        )
        self.q_layernorm = Qwen4ExpTextRMSNorm(self.index_head_dim, eps=config.rms_norm_eps)
        self.k_layernorm = Qwen4ExpTextRMSNorm(self.index_head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor,
        past_key_values: Cache | None,
    ) -> torch.Tensor:
        batch_size, seq_length, _ = hidden_states.shape
        hidden_shape = (batch_size, seq_length, -1, self.index_head_dim)
        # The cos/sin here are the full positions for the keys, so we need to slice to get only the current positions for the queries
        full_cos, full_sin = position_embeddings
        current_cos, current_sin = full_cos[:, -seq_length:, :], full_sin[:, -seq_length:, :]

        qk = self.index_qk_proj(hidden_states)
        q, token_k = torch.split(
            qk,
            [self.index_n_heads * self.index_head_dim, self.index_kv_heads * self.index_head_dim],
            dim=-1,
        )
        q, raw_keys = q.reshape(*hidden_shape), token_k.reshape(*hidden_shape).squeeze(2)
        q = self.q_layernorm(q)
        q = apply_rotary_pos_emb(q, cos=current_cos, sin=current_sin, unsqueeze_dim=2)

        if past_key_values is not None:
            raw_keys = past_key_values.update_indexer(raw_keys, self.layer_idx)

        # Note that the mask is never None here as we only allow eager and sdpa, and we do not allow sdpa's mask skip
        # It's always 4D with either bool (sdpa) or float (eager) and already gives us the valid indices
        visible_token_indices = attention_mask if attention_mask.dtype == torch.bool else attention_mask == 0

        selected_token_indices = torch.full(
            (batch_size, seq_length, self.token_budget + self.compress_ratio - 1),
            -1,
            dtype=torch.int32,
            device=hidden_states.device,
        )
        for batch_idx in range(batch_size):
            for query_idx in range(seq_length):
                local_visible_indices = torch.nonzero(
                    visible_token_indices[batch_idx, 0, query_idx], as_tuple=False
                ).flatten()
                num_complete_blocks = local_visible_indices.shape[-1] // self.compress_ratio
                # Compute selected tokens
                if num_complete_blocks > 0:
                    block_token_indices = local_visible_indices[: num_complete_blocks * self.compress_ratio].view(
                        num_complete_blocks, self.compress_ratio
                    )

                    key_groups = raw_keys[batch_idx].index_select(0, block_token_indices.flatten())
                    key_groups = key_groups.view(*block_token_indices.shape, self.index_head_dim)
                    pooled_keys = key_groups.float().mean(dim=1).to(raw_keys.dtype)
                    pooled_keys = self.k_layernorm(pooled_keys)
                    group_starts = block_token_indices[:, 0]
                    block_key_states = apply_rotary_pos_emb(
                        pooled_keys.unsqueeze(1),
                        cos=full_cos[batch_idx].index_select(0, group_starts),
                        sin=full_sin[batch_idx].index_select(0, group_starts),
                    ).squeeze(1)

                    scores = torch.matmul(
                        q[batch_idx, query_idx].float(), block_key_states.float().transpose(-1, -2)
                    ).transpose(-1, -2)
                    scores = torch.relu(scores).sum(dim=-1) / math.sqrt(self.index_head_dim)

                    selected_block_indices = scores.topk(min(self.block_topk, num_complete_blocks), dim=0).indices
                    # Remap the indices of the blocks to the indices of individual tokens
                    selected_tokens = block_token_indices.index_select(0, selected_block_indices).flatten()
                else:
                    selected_tokens = torch.tensor([], device=hidden_states.device)
                tail = local_visible_indices[num_complete_blocks * self.compress_ratio :]
                selected_tokens = torch.cat([selected_tokens, tail]).to(torch.int32)
                selected_token_indices[batch_idx, query_idx, : selected_tokens.numel()] = selected_tokens

        # Create the additive mask to be added to the main causal mask
        kv_length = attention_mask.shape[-1]
        selected_token_mask = torch.zeros(
            (*selected_token_indices.shape[:-1], kv_length + 1), device=attention_mask.device, dtype=torch.bool
        )
        # We absorb all the -1 by scaterring them to the last index that we will drop
        scatter_indices = torch.where(selected_token_indices >= 0, selected_token_indices, kv_length)
        selected_token_mask = selected_token_mask.scatter(-1, scatter_indices, True)[..., :kv_length].unsqueeze(1)
        # if using eager, convert to float mask
        if attention_mask.is_floating_point():
            min_dtype = torch.finfo(attention_mask.dtype).min
            selected_token_mask = torch.where(selected_token_mask, attention_mask.new_zeros(()), min_dtype)

        return selected_token_mask


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class Qwen4ExpTextAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Qwen4ExpTextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim * 2, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = Qwen4ExpTextRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen4ExpTextRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.indexer = Qwen4ExpTextQSAIndexer(config, layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        selected_token_mask = self.indexer(hidden_states, position_embeddings, attention_mask, past_key_values)
        # Combine both masks (they are never None, and are always 4D with either bool for sdpa, or float for eager)
        if attention_mask.is_floating_point():
            attention_mask = attention_mask + selected_token_mask
        else:
            attention_mask = attention_mask & selected_token_mask

        # The cos/sin are the full positions here due to the indexer, so we need to slice to get current positions
        position_embeddings = (x[:, -hidden_states.shape[1] :, :] for x in position_embeddings)
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states, gate = torch.chunk(
            self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1
        )
        gate = gate.reshape(*input_shape, -1)

        query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)

        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Qwen4ExpTextMLP(nn.Module):
    def __init__(self, config, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class Qwen4ExpTextExperts(nn.Module):
    """Collection of expert weights stored as 3D tensors."""

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts + 1)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate, up = nn.functional.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = nn.functional.linear(current_hidden_states, self.down_proj[expert_idx])
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states


class Qwen4ExpTextTopKRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.norm_topk_prob = config.norm_topk_prob
        self.hidden_dim = config.hidden_size
        self.weight = nn.Parameter(torch.zeros(self.num_experts, self.hidden_dim))

    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states, self.weight)  # (seq_len, num_experts)
        router_probs = torch.nn.functional.softmax(router_logits, dtype=torch.float, dim=-1)
        router_top_value, router_indices = torch.topk(router_probs, self.top_k, dim=-1)  # (seq_len, top_k)
        if self.norm_topk_prob:
            router_top_value /= router_top_value.sum(dim=-1, keepdim=True)
        router_top_value = router_top_value.to(router_logits.dtype)
        router_scores = router_top_value
        return router_logits, router_scores, router_indices


class Qwen4ExpTextSparseMoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate = Qwen4ExpTextTopKRouter(config)
        self.experts = Qwen4ExpTextExperts(config)
        self.shared_expert = Qwen4ExpTextMLP(config, intermediate_size=config.shared_expert_intermediate_size)
        self.shared_expert_gate = torch.nn.Linear(config.hidden_size, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)
        shared_expert_output = self.shared_expert(hidden_states_reshaped)
        _, routing_weights, selected_experts = self.gate(hidden_states_reshaped)
        expert_output = self.experts(hidden_states_reshaped, selected_experts, routing_weights)

        shared_expert_output = F.sigmoid(self.shared_expert_gate(hidden_states_reshaped)) * shared_expert_output

        expert_output = expert_output + shared_expert_output
        expert_output = expert_output.reshape(batch_size, sequence_length, hidden_dim)
        return expert_output


class Qwen4ExpTextGatedResidual(nn.Module):
    def __init__(self, config: Qwen4ExpTextConfig, use_combine: bool = True):
        super().__init__()
        self.hc_count = config.hc_count
        self.hidden_size = config.hidden_size
        hc_hidden_size = self.hc_count * self.hidden_size
        self.hc_norm = Qwen4ExpTextRMSNorm(hc_hidden_size, group_size=self.hidden_size, eps=config.rms_norm_eps)
        self.input_mix_weight_down = nn.Linear(hc_hidden_size, config.hc_lowrank, bias=False)
        self.input_mix_weight_up = nn.Linear(config.hc_lowrank, hc_hidden_size, bias=False)
        self.block_inject_weight = nn.Linear(hc_hidden_size, self.hc_count, bias=False) if use_combine else None

    def forward(
        self, hyper_input: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if hyper_input.shape[-1] != self.hc_count * self.hidden_size:
            raise ValueError(
                f"Expected {self.hc_count * self.hidden_size} hyper-connection features, got {hyper_input.shape[-1]}."
            )
        hyper_input_normed = self.hc_norm(hyper_input)
        input_mix_weight = F.silu(self.input_mix_weight_down(hyper_input_normed) / self.hc_count)
        input_mix_weight = torch.sigmoid(self.input_mix_weight_up(input_mix_weight))
        input_mix_weight = input_mix_weight.unflatten(-1, (self.hc_count, self.hidden_size))
        mixed_input = (input_mix_weight * hyper_input_normed.unflatten(-1, (self.hc_count, self.hidden_size))).mean(
            dim=-2
        )
        if self.block_inject_weight is None:
            return mixed_input
        injection_weights = 2 * torch.sigmoid(self.block_inject_weight(hyper_input_normed) / self.hc_count)
        return mixed_input, hyper_input, injection_weights


_MASK64 = (1 << 64) - 1
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_M1 = 0xBF58476D1CE4E5B9
_SPLITMIX_M2 = 0x94D049BB133111EB
_PRIME_1 = 10007


def _splitmix64(value: int) -> int:
    value = (value + _SPLITMIX_GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * _SPLITMIX_M1) & _MASK64
    value = ((value ^ (value >> 27)) * _SPLITMIX_M2) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def _build_layer_multipliers(unigram_vocab_size, ngram_size, ple_layer_index, seed: int) -> torch.Tensor:
    max_long = (1 << 63) - 1
    multiplier_max = max_long // max(unigram_vocab_size, 1)
    half_bound = max(1, multiplier_max // 2)
    base_seed = seed + _PRIME_1 * ple_layer_index
    multipliers = []
    for index in range(ngram_size):
        value = (base_seed + _SPLITMIX_GAMMA * (index + 1)) & _MASK64
        multipliers.append(2 * (_splitmix64(value) % half_bound) + 1)
    return torch.tensor(multipliers, dtype=torch.long)


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    for divisor in range(3, math.isqrt(value) + 1, 2):
        if value % divisor == 0:
            return False
    return True


def _find_nth_prime_after(start: int, count: int) -> int:
    prime = start
    for _ in range(count):
        prime += 1
        while not _is_prime(prime):
            prime += 1
    return prime


class Qwen4ExpTextNGramEmbedding(nn.Module):
    def __init__(self, config: Qwen4ExpTextConfig, embedding_dim: int, layer_idx: int, ple_layer_index: int = 0):
        super().__init__()
        self.layer_idx = layer_idx
        self.ngram_size = config.ngram_size
        self.context_len = self.ngram_size - 1
        self.heads_per_ngram = config.heads_per_ngram
        self.ngram_heads = (self.ngram_size - 1) * self.heads_per_ngram
        self.ple_layer_index = ple_layer_index
        self.unigram_vocab_size = config.vocab_size
        self.ngram_vocab_size_base = config.ngram_vocab_size_base
        head_dim_per_ngram = embedding_dim // self.ngram_heads
        self.seed = config.seed
        # CODEPATH: @ArthurZucker fix flagging for no reason here
        self.eos_token_id = config.eos_token_id[0] if isinstance(config.eos_token_id, list) else config.eos_token_id

        self.head_vocab_sizes = []
        self.head_offsets = []
        self.total_vocab_size = 0
        for head_idx in range(self.ngram_heads):
            global_head_idx = self.ple_layer_index * self.ngram_heads + head_idx
            size = _find_nth_prime_after(self.ngram_vocab_size_base - 1, global_head_idx + 1)
            self.head_vocab_sizes.append(size)
            self.head_offsets.append(self.total_vocab_size)
            self.total_vocab_size += size

        self.layer_multipliers = nn.Buffer(
            _build_layer_multipliers(self.unigram_vocab_size, self.ngram_size, self.ple_layer_index, self.seed)
        )
        self.ngram_heads_vocab_sizes = nn.Buffer(torch.tensor(self.head_vocab_sizes, dtype=torch.long))
        self.ngram_heads_offsets = nn.Buffer(torch.tensor(self.head_offsets, dtype=torch.long))
        ngram_vocab_divisor = config.make_ngram_vocab_size_divisible_by
        padded_vocab_size = math.ceil(self.total_vocab_size / ngram_vocab_divisor) * ngram_vocab_divisor
        self.ngram_embedding = nn.Embedding(padded_vocab_size, head_dim_per_ngram)

    def _shift_right_ignore_eos(self, token_ids: torch.Tensor, shift: int) -> torch.Tensor:
        if shift == 0:
            return token_ids
        batch_size, seq_len = token_ids.shape
        positions = torch.arange(seq_len, device=token_ids.device, dtype=torch.long)
        eos_positions = torch.where(token_ids == self.eos_token_id, positions, -1)
        previous_eos_inclusive = torch.cummax(eos_positions, dim=1).values
        previous_eos = torch.cat([eos_positions.new_full((batch_size, 1), -1), previous_eos_inclusive[:, :-1]], dim=1)
        segment_start = previous_eos + 1
        position_in_segment = positions.unsqueeze(0) - segment_start
        source_positions = positions - shift
        gather_positions = source_positions.clamp_min(0).unsqueeze(0).expand(batch_size, -1)
        shifted = token_ids.gather(dim=1, index=gather_positions)
        valid = (position_in_segment >= shift) & (source_positions.unsqueeze(0) >= 0)
        return torch.where(valid, shifted, token_ids.new_full((), self.eos_token_id))

    def forward(self, input_ids: torch.Tensor, past_key_values: Cache | None) -> torch.Tensor:
        input_ids = input_ids.long()
        # This is a trick to store the previous N=self.context_len `input_ids` - indeed the manipulations are identical to storing
        # a past conv_state, so we can use an additional conv_states inside the Cache for it
        if past_key_values is not None and past_key_values.has_previous_state(self.layer_idx, state_idx=2):
            previous_context = past_key_values.layers[self.layer_idx].conv_states[2].clone()
        else:
            previous_context = input_ids.new_full((input_ids.shape[0], self.context_len), self.eos_token_id)
        # Store the current input_ids for the next forward
        if past_key_values is not None:
            input_ids_to_cache = input_ids
            # In the case where `input_ids` would be smaller than `self.context_len`, the `update_conv_state` will pad with zeros, whereas
            # here we want to pad with eos, so we do it explicitly
            if (
                not past_key_values.has_previous_state(self.layer_idx, state_idx=2)
                and input_ids.shape[1] < self.context_len
            ):
                input_ids_to_cache = torch.nn.functional.pad(
                    input_ids_to_cache, (self.context_len - input_ids.shape[1], 0), value=self.eos_token_id
                )
            _ = past_key_values.update_conv_state(
                input_ids_to_cache, self.layer_idx, state_idx=2, conv_kernel_size=self.context_len
            )

        # Get full token history
        token_history = torch.cat([previous_context, input_ids], dim=-1)
        shifted_tokens = [self._shift_right_ignore_eos(token_history, shift) for shift in range(self.ngram_size)]

        blocks = []
        for ngram in range(2, self.ngram_size + 1):
            start_idx = (ngram - 2) * self.heads_per_ngram
            end_idx = start_idx + self.heads_per_ngram
            mixed_ids = shifted_tokens[0] * self.layer_multipliers[0]
            for position in range(1, ngram):
                mixed_ids = torch.bitwise_xor(
                    mixed_ids,
                    shifted_tokens[position] * self.layer_multipliers[position],
                )
            head_vocab_sizes = self.ngram_heads_vocab_sizes[start_idx:end_idx]
            head_offsets = self.ngram_heads_offsets[start_idx:end_idx]
            ngram_ids = torch.remainder(mixed_ids.unsqueeze(-1), head_vocab_sizes.view(1, 1, -1))
            blocks.append(ngram_ids + head_offsets.view(1, 1, -1))

        ngram_ids = torch.cat(blocks, dim=-1)[:, -input_ids.shape[1] :]
        # We need explicit device placement here, as the embedding may be skipped from device_map completely (we just need to be
        # careful in the case of offloading to disk)
        execution_device = (
            self.ngram_embedding.weight.device if self.ngram_embedding.weight.device.type != "meta" else None
        )
        return self.ngram_embedding(ngram_ids.to(execution_device)).to(ngram_ids.device).flatten(-2)


class Qwen4ExpTextPLELayer(nn.Module):
    """Inject hashed n-gram features into every hyper-connection stream.

    PLE projects each token's concatenated n-gram embedding to a shared value and one key per residual stream. The
    normalized stream activations gate those values, then a dilated depthwise convolution adds local lexical context.
    The returned tensor has shape `(batch_size, sequence_length, hc_count * hidden_size)`.
    """

    def __init__(self, config: Qwen4ExpTextConfig, layer_idx: int, ple_layer_index: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.hc_count = config.hc_count
        ple_embed_dim = config.ple_embed_dim
        hc_hidden_size = self.hidden_size * self.hc_count
        self.ple_embedding = Qwen4ExpTextNGramEmbedding(config, ple_embed_dim, layer_idx, ple_layer_index)
        conv_kernel_size = config.ple_conv_kernel_size
        conv_dilation = config.ngram_size
        self.short_conv_state_len = (conv_kernel_size - 1) * conv_dilation
        self.key_proj = nn.Linear(ple_embed_dim, hc_hidden_size, bias=False)
        self.value_proj = nn.Linear(ple_embed_dim, self.hidden_size, bias=False)
        self.norm_key = Qwen4ExpTextRMSNorm(hc_hidden_size, group_size=self.hidden_size, eps=config.rms_norm_eps)
        self.norm_query = Qwen4ExpTextRMSNorm(hc_hidden_size, group_size=self.hidden_size, eps=config.rms_norm_eps)
        self.norm_conv = Qwen4ExpTextRMSNorm(hc_hidden_size, group_size=self.hidden_size, eps=config.rms_norm_eps)
        self.conv1d = nn.Conv1d(
            hc_hidden_size,
            hc_hidden_size,
            kernel_size=conv_kernel_size,
            groups=hc_hidden_size,
            dilation=conv_dilation,
            bias=False,
        )

    def _short_conv(self, hidden_states: torch.Tensor, past_key_values: Cache | None) -> torch.Tensor:
        seq_len = hidden_states.shape[1]
        hidden_states = hidden_states.transpose(1, 2)

        if past_key_values is not None:
            hidden_states = past_key_values.update_conv_state(
                hidden_states, self.layer_idx, state_idx=1, conv_kernel_size=self.short_conv_state_len
            )

        # We always pad and slice due to the dilation in the conv, to make sure we have enough states
        hidden_states = F.pad(hidden_states, (self.short_conv_state_len, 0))
        hidden_states = hidden_states[..., -(self.short_conv_state_len + seq_len) :]

        # We cannot use the usual functions/kernels here for the short conv as the conv1d has dilation
        hidden_states = F.silu(self.conv1d(hidden_states))

        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        past_key_values: Cache | None,
        conv_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        embeddings = self.ple_embedding(input_ids, past_key_values)
        key_normed = self.norm_key(self.key_proj(embeddings)).unflatten(-1, (self.hc_count, self.hidden_size))
        value = self.value_proj(embeddings)
        query_normed = self.norm_query(hidden_states).unflatten(-1, (self.hc_count, self.hidden_size))
        gate = (key_normed * query_normed).sum(dim=-1, keepdim=True) / math.sqrt(self.hidden_size)
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        gated_value = torch.sigmoid(gate) * value.unsqueeze(-2)
        gated_value_normed = self.norm_conv(gated_value.flatten(-2))
        gated_value = gated_value.flatten(-2)
        if conv_mask is not None:
            gated_value = apply_mask_to_padding_states(gated_value, conv_mask)
            gated_value_normed = apply_mask_to_padding_states(gated_value_normed, conv_mask)
        output = gated_value + self._short_conv(gated_value_normed, past_key_values)
        return output


class Qwen4ExpTextDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen4ExpTextConfig, layer_idx: int):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx]
        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen4ExpTextGatedDeltaNet(config, layer_idx)
        else:
            self.self_attn = Qwen4ExpTextAttention(config, layer_idx)
        self.mlp = Qwen4ExpTextSparseMoeBlock(config)
        # CODEPATH: @ArthurZucker fix flagging for no reason here
        ple_layer_index = config.ple_layer_ids.index(layer_idx + 1) if layer_idx + 1 in config.ple_layer_ids else None
        self.ple = Qwen4ExpTextPLELayer(config, layer_idx, ple_layer_index) if ple_layer_index is not None else None
        self.attn_hyper_connection = Qwen4ExpTextGatedResidual(config)
        self.mlp_hyper_connection = Qwen4ExpTextGatedResidual(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        conv_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        ple_input_ids: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.FloatTensor:
        if self.ple is not None:
            hidden_states = hidden_states + self.ple(hidden_states, ple_input_ids, past_key_values, conv_mask=conv_mask)

        hidden_states, hyper_input, injection_weights = self.attn_hyper_connection(hidden_states)
        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(
                hidden_states, cache_params=past_key_values, attention_mask=conv_mask, **kwargs
            )
        else:
            hidden_states, _ = self.self_attn(
                hidden_states,
                position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                **kwargs,
            )

        injection = hidden_states.unsqueeze(-2) * injection_weights.unsqueeze(-1)
        hidden_states = hyper_input + injection.flatten(-2)

        hidden_states, hyper_input, injection_weights = self.mlp_hyper_connection(hidden_states)
        hidden_states = self.mlp(hidden_states)

        injection = hidden_states.unsqueeze(-2) * injection_weights.unsqueeze(-1)
        hidden_states = hyper_input + injection.flatten(-2)
        return hidden_states
