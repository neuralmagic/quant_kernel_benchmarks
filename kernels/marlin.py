import torch
from typing import Callable, Optional

from torch.utils.benchmark import Measurement as TMeasurement
from weight_shapes import WEIGHT_SHAPES

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    GPTQ_MARLIN_MAX_PARALLEL, GPTQ_MARLIN_MIN_THREAD_N, marlin_permute_scales,
    marlin_zero_points)
from vllm.model_executor.layers.quantization.utils.marlin_utils_test import (
    MarlinWorkspace)
from vllm.scalar_type import scalar_types

from .utils import BenchmarkTensors

def create_bench_fn(bt: BenchmarkTensors, 
                    out_type: Optional[torch.dtype]) -> Callable:
    device = bt.a.device

    workspace = MarlinWorkspace(bt.w_ref.shape[1], GPTQ_MARLIN_MIN_THREAD_N,
                                GPTQ_MARLIN_MAX_PARALLEL)

    if bt.w_g_zp is None:
        w_zp = torch.empty(0, dtype=torch.int, device=device)
    else:
        w_zp = marlin_zero_points(
            bt.w_g_zp, bt.w_ref.shape[0], bt.w_ref.shape[1], bt.wtype.size_bits)
        
    if bt.group_size is None:
        w_s = torch.tensor([], device="cuda", dtype=torch.half)
    else:
        w_s = marlin_permute_scales(
            bt.w_g_s,  bt.w_ref.shape[0], bt.w_ref.shape[1], bt.group_size)

    sort_indices = torch.empty(0, dtype=torch.int, device=device)
    g_idx = torch.empty(0, dtype=torch.int, device=device)
    w_q = ops.gptq_marlin_repack(
        bt.w_q_packed.contiguous(), sort_indices, bt.w_ref.shape[0], 
        bt.w_ref.shape[1], bt.wtype.size_bits)

    if bt.a.dtype.is_floating_point:
        assert bt.w_ch_s is None
        assert bt.w_tok_s is None
        assert bt.group_size is not None

        fn = lambda: ops.gptq_marlin_gemm(a=bt.a,
                                          b_q_weight=w_q,
                                          b_scales=w_s,
                                          b_zeros=w_zp,
                                          g_idx=g_idx,
                                          perm=sort_indices,
                                          workspace=workspace.scratch,
                                          b_q_type=bt.wtype,
                                          size_m=bt.a.shape[0],
                                          size_n=bt.w_ref.shape[1],
                                          size_k=bt.w_ref.shape[0],
                                          is_k_full=True)
    else:
        assert bt.a.dtype == torch.int8
        assert bt.wtype == scalar_types.uint4b8
        
        if bt.w_ch_s is not None:
            s_ch = bt.w_ch_s.to(torch.float32)
        else:    
            s_ch = torch.ones(bt.w_ref.shape[1],
                          dtype=torch.float32,
                          device=device)
        
        if bt.w_tok_s is not None:
            s_tok = bt.w_tok_s.to(torch.float32)
        else:
            s_tok = torch.ones(
                bt.a.shape[0], dtype=torch.float32, device=device)

        fn = lambda: ops.marlin_qqq_gemm(a=bt.a,
                                         b_q_weight=w_q,
                                         s_group=w_s,
                                         s_tok=s_tok,
                                         s_ch=s_ch,
                                         workspace=workspace.scratch,
                                         size_m=bt.a.shape[0],
                                         size_n=bt.w_ref.shape[1],
                                         size_k=bt.w_ref.shape[0])

    return fn
