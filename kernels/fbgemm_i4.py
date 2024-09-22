import sys

try:
    import fbgemm_gpu.experimental.gen_ai  # noqa: F401
except ImportError:
    print("fbgemm kernel not found, please install:")
    print(" pip install fbgemm_gpu")
    sys.exit()

from typing import Tuple, Optional, Callable
import torch

from .utils import BenchmarkTensors


def create_bench_fn(bt: BenchmarkTensors,
                    out_type: Optional[torch.dtype],
                    schedule=None) -> Callable:
    
    assert bt.a.dtype == torch.bfloat16
    
    w_q = bt.w_q_packed.t().contiguous()
    
    zeros = bt.w_g_zp if bt.w_g_zp is not None else torch.zeros_like(bt.w_g_s)
    return lambda: torch.ops.fbgemm.bf16i4bf16_rowwise(
        bt.a, w_q, bt.w_g_s, zeros)
