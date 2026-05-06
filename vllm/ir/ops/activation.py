# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch.nn.functional as F
from torch import Tensor

from ..op import register_op


@register_op
def silu_and_mul(x: Tensor) -> Tensor:
    """SwiGLU activation: silu(x[:d]) * x[d:]

    Computes the SwiGLU gated activation function where the input is split
    along the last dimension into gate and up projections.

    Shapes:
        x: (..., 2 * d)
        return: (..., d)
    """
    d = x.shape[-1] // 2
    return F.silu(x[..., :d]) * x[..., d:]


@silu_and_mul.register_input_generator
def _silu_and_mul_input_generator(
    num_tokens: int, hidden_size: int, dtype: torch.dtype
) -> tuple:
    x = torch.randn(num_tokens, hidden_size * 2, dtype=dtype)
    return (x,)



