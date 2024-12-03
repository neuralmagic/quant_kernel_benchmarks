import torch
from typing import Callable, Optional

from vllm import _custom_ops as ops

from .utils import BenchmarkTensors

BARRIER_WORKSPACE = torch.zeros(16384, dtype=torch.uint8, device="cuda")

def create_bench_fn(bt: BenchmarkTensors,
                    out_type:  Optional[torch.dtype],
                    schedule=None) -> Callable:
    w_q = bt.w_q.t().contiguous().t()  # make col major
    # Convert to,
    w_q = ops.machete_prepack_B(w_q, bt.a.dtype, bt.wtype, 
                            None if bt.w_g_s is None else bt.w_g_s.dtype)
    # once https://github.com/vllm-project/vllm/pull/8046 lands
    # w_q = ops.machete_prepack_B(w_q, bt.wtype)

    w_g_zp = bt.w_g_zp
    if w_g_zp is not None:
        w_g_zp = -1 * bt.w_g_s * (w_g_zp.to(bt.w_g_s.dtype))

    # Convert to,
    return lambda: ops.machete_mm(
        a=bt.a,
        b_q=bt.w_q,
        b_type=bt.wtype,
        b_group_scales=bt.w_g_s,
        b_group_zeros=w_g_zp,
        b_group_size=bt.group_size,
        b_channel_scales=bt.w_ch_s,
        a_token_scales=bt.w_tok_s,
        out_type=out_type,
        #barrier_workspace=BARRIER_WORKSPACE,
        schedule=schedule,
    )
    # once https://github.com/vllm-project/vllm/pull/8046 lands
    # return lambda: ops.machete_gemm(
    #     a=bt.a,
    #     b_q=bt.w_q_packed,
    #     b_type=bt.wtype,
    #     b_scales=bt.w_g_s,
    #     b_zeros=w_g_zp,
    #     b_group_size=bt.group_size,
    #     schedule=schedule,
    # )
