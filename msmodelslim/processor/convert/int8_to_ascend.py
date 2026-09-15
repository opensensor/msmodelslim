"""Lossless weight export from compressed-tensors INT8 to ModelSlim W8A8_DYNAMIC."""

from __future__ import annotations

import torch

from msmodelslim.core.convert.types import IRKind, LossLevel
from msmodelslim.core.quant_service.modelslim_convert.virtual_module import ModelFreeLinear
from msmodelslim.ir import W8A8DynamicPerChannelFakeQuantLinear, int8_per_channel_sym, int8_per_token_sym
from msmodelslim.ir.qal import QDType, QParam, QStorage
from msmodelslim.processor.convert.base import BaseConvertProcessor


class Int8ToAscendProcessor(BaseConvertProcessor):
    name = "Int8ToAscendProcessor"
    src_ir = IRKind.INT8_PER_CHANNEL
    dst_ir = IRKind.W8A8_DYNAMIC
    loss_level = LossLevel.LOSSLESS.value

    def transform(self, module, context):
        if not isinstance(module, ModelFreeLinear):
            raise TypeError("INT8 import requires a ModelFreeLinear")
        if not module.lazy_initialized:
            module.lazy_init(context.reader, device=context.resolved_worker_device)
        weight = getattr(module, "weight", None)
        scale = getattr(module, "weight_scale", None)
        if weight is None or weight.dtype != torch.int8 or weight.ndim != 2:
            raise ValueError(f"{module.full_name}: expected a 2D INT8 weight")
        if (
            scale is None
            or scale.dtype not in (torch.float32, torch.float16, torch.bfloat16)
            or scale.shape not in ((weight.shape[0],), (weight.shape[0], 1))
        ):
            raise ValueError(f"{module.full_name}: expected floating-point per-output-channel scales")
        if not torch.isfinite(scale).all() or not (scale > 0).all():
            raise ValueError(f"{module.full_name}: weight scales must be finite and positive")
        zero_point = getattr(module, "weight_zero_point", None)
        if zero_point is not None and (zero_point.shape != scale.shape or torch.count_nonzero(zero_point)):
            raise ValueError(f"{module.full_name}: only symmetric zero-point-free weights are supported")
        bias = getattr(module, "bias", None)
        if bias is not None and (
            bias.shape != (weight.shape[0],) or bias.dtype not in (torch.float32, torch.float16, torch.bfloat16)
        ):
            raise ValueError(f"{module.full_name}: invalid linear bias")
        # Reuse the native IR and AscendV1Saver; never dequantize/requantize.
        # The saver expands scales to [out, 1] and supplies zero weight_offset.
        return W8A8DynamicPerChannelFakeQuantLinear(
            x_q_param=QParam(scheme=int8_per_token_sym),
            w_q_param=QParam(scheme=int8_per_channel_sym, ext={"scale": scale.float().reshape(-1)}),
            w_q=QStorage(dtype=QDType.INT8, value=weight.detach()),
            bias=bias,
        )
