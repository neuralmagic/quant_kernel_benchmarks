from typing import Callable, Iterable, List, Optional, Tuple
from dataclasses import dataclass
import torch
import math
import torch.utils.benchmark as TBenchmark

from vllm.model_executor.layers.quantization.utils.quant_utils import (
    quantize_weights, pack_quantized_values_into_int32)
from vllm.scalar_type import ScalarType

def terse_type_name(dt):
    return {
        torch.bfloat16: "bf16",
        torch.float16: "fp16",
        torch.int8: "int8",
        torch.float8_e4m3fn: "fp8",
        torch.bfloat16: "bf16",
        torch.float: "float",
        torch.int: "int",
    }[dt]


@dataclass
class BenchmarkTensors:
    w_ref: torch.Tensor
    a: torch.Tensor

    w_q: torch.Tensor
    w_q_packed: torch.Tensor
    group_size: Optional[int]
    wtype: ScalarType
    w_g_s: torch.Tensor
    w_g_zp: Optional[torch.Tensor]
    w_ch_s: Optional[torch.Tensor]
    w_tok_s: Optional[torch.Tensor]

@dataclass
class TypeConfig:
    act_type: torch.dtype
    weight_type: ScalarType
    output_type: Optional[torch.dtype]
    group_scale_type: Optional[torch.dtype]
    group_zero_type: Optional[torch.dtype]
    channel_scale_type: Optional[torch.dtype]
    token_scale_type: Optional[torch.dtype]

def rand_data(shape, dtype=torch.float16, scale=1):
    if dtype.is_floating_point:
        return (scale * torch.rand(shape, device="cuda") - 0.3).to(dtype)
    else:
        return torch.randint(-15, 15, shape, dtype=dtype, device="cuda")


def quantize_and_pack(atype: torch.dtype,
                      w: torch.Tensor,
                      wtype: ScalarType,
                      stype: Optional[torch.dtype],
                      group_size: Optional[int],
                      zero_points: bool = False):
    assert wtype.is_integer(), "TODO: support floating point weights"

    w_ref, w_q, w_s, w_zp = quantize_weights(
        w,
        wtype,
        group_size=group_size,
        zero_points=zero_points,
        # to match how the kernel applies zps
        ref_zero_points_after_scales=True)

    w_q_packed = pack_quantized_values_into_int32(w_q, wtype)
    return w_ref, w_q, w_q_packed, w_s, w_zp


def create_bench_tensors(shape: Tuple[int, int, int],
                        types: TypeConfig,
                        group_size: Optional[int]) -> List[BenchmarkTensors]:
    m, n, k = shape
    
    # we want to make sure that weights don't fit into L2 cache between runs so
    #  we construct enough weights to exceed L2 cache, which is 50mb on a H100
    #  so we target total weight size > 2*50mb
    num_weights = math.ceil(2 * 50 * 1024**2 * 8 / 
                            (k * n * types.weight_type.size_bits))

    a = rand_data((m, k), types.act_type, scale=5)
    
    benchmark_tensors: List[BenchmarkTensors] = []
    for _ in range(num_weights):
        w = rand_data((k, n), types.act_type, scale=5)

        if types.group_scale_type is not None:
            w = w.to(types.group_scale_type)
        if w.dtype.itemsize == 1:
            w = w.to(torch.float16)

        w_ref, w_q, w_q_packed, w_s, w_zp = quantize_and_pack(
            a.dtype, w, types.weight_type, types.group_scale_type, group_size,
            types.group_zero_type is not None)

        if not a.dtype.is_floating_point:
            aiinfo = torch.iinfo(a.dtype)
            w_ref = w_ref.round().clamp(aiinfo.min, aiinfo.max)

        w_ref = w_ref.to(torch.float32)

        w_ch_s = None if types.channel_scale_type is None else\
            rand_data((n,), types.channel_scale_type)
        w_tok_s = None if types.token_scale_type is None else\
            rand_data((m,), types.token_scale_type)
            
        benchmark_tensors.append(
            BenchmarkTensors(w_ref=w_ref,
                   a=a,
                   w_q=w_q,
                   w_q_packed=w_q_packed,
                   wtype=types.weight_type,
                   w_g_s=w_s,
                   w_g_zp=w_zp,
                   group_size=group_size,
                   w_ch_s=w_ch_s,
                   w_tok_s=w_tok_s)
        )
        
    return benchmark_tensors
