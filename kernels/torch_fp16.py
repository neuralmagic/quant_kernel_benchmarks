
import torch
from typing import Callable

from .utils import BenchmarkTensors, TypeConfig

def name(t: TypeConfig):
    return "torch.mm (fp16)"

def create_bench_fn(bt: BenchmarkTensors, out_type: torch.dtype) -> Callable:
    a, w = bt.a, bt.w_ref.to(bt.a.dtype)
    if a.dtype not in [torch.float16, torch.bfloat16]:
        a = a.to(torch.float16)
        w = w.to(torch.float16)
    return lambda: torch.matmul(a, w)
