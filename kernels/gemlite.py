try:
    import gemlite
except ImportError:
    print("Gemlite kernel not found, please install:")
    print(" pip install git+https://github.com/mobiusml/gemlite/")

from gemlite.core import GemLiteLinearTriton, DType

import torch
from typing import Callable, Optional

from .utils import BenchmarkTensors

def create_bench_fn(bt: BenchmarkTensors,
                    out_type: Optional[torch.dtype],
                    schedule=None) -> Callable:
    
    def torch_dtype_to_gemmlite_dtype(t: torch.dtype) -> DType:
        return{
            torch.float32: DType.FP32,
            torch.float16: DType.FP16,
            torch.int8: DType.INT8,
            torch.bfloat16: DType.BF16
        }[t]
    
    input_dtype = torch_dtype_to_gemmlite_dtype(bt.a.dtype)
    if out_type is None:
        output_type = input_dtype
    else:
        output_type = torch_dtype_to_gemmlite_dtype(out_type)
    
    gemlite_linear = GemLiteLinearTriton(W_nbits=bt.wtype.size_bits, 
                                         group_size=bt.group_size, 
                                         in_features=bt.w_ref.shape[0], 
                                         out_features=bt.w_ref.shape[1], 
                                         input_dtype=input_dtype, 
                                         output_dtype=output_type)

    zeros = bt.w_g_zp
    if zeros is None:
        zeros = torch.zeros_like(bt.w_g_s)
    gemlite_linear.pack(
        bt.w_q, 
        bt.w_g_s, 
        zeros,
        bt.w_ch_s
    )

    return lambda: gemlite_linear.forward_auto(bt.a)
    
