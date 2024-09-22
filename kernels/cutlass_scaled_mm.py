import torch
from typing import Callable, Optional
from vllm import _custom_ops as ops

from .utils import BenchmarkTensors, TypeConfig

def can_run_for_types(t: TypeConfig):
    return t.act_type in [torch.int8, torch.float8_e4m3fn]


def create_bench_fn(bt: BenchmarkTensors, 
                    out_type:  Optional[torch.dtype]) -> Callable:
    if bt.w_ch_s is not None and bt.w_tok_s is not None:
        scale_a = bt.w_tok_s.to(torch.float32)
        scale_b = bt.w_ch_s.to(torch.float32)
    else:
        scale_a = torch.tensor(1.0, dtype=torch.float32, device=bt.a.device)
        scale_b = torch.tensor(1.0, dtype=torch.float32, device=bt.a.device)
    w_col_major = bt.w_ref.to(bt.a.dtype).t().contiguous().t()
    return lambda: ops.cutlass_scaled_mm(
        bt.a, w_col_major, scale_a, scale_b, out_dtype=torch.float16)