import argparse
import copy
import os
import itertools
import math
import pickle as pkl
import time
import torch
import torch.utils.benchmark as TBenchmark
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Tuple

from torch.utils.benchmark import Measurement as TMeasurement
from weight_shapes import WEIGHT_SHAPES

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    pack_rows, quantize_weights)
from vllm.scalar_type import ScalarType, scalar_types
from vllm.utils import FlexibleArgumentParser

from kernels import get_kernel_module
from kernels.utils import BenchmarkTensors, TypeConfig, terse_type_name, create_bench_tensors

DEFAULT_MODELS = ["meta-llama/Llama-3-8b", "meta-llama/Llama-2-70b-hf"]
DEFAULT_BATCH_SIZES = [1, 16, 32, 64, 128, 256, 512, 1024]
DEFAULT_TP_SIZES = [1]

NVTX_PROFILE = os.environ.get("NVTX_PROFILE", False)

if NVTX_PROFILE:
    import nvtx

# impl

# bench

def bench_fns(label: str, sub_label: str, description: str,
              fns: List[Callable]):
    
    min_run_time = 1 if not NVTX_PROFILE else 0.1
    res = TBenchmark.Timer(
        stmt="""
        for fn in fns:
            fn()
        """,
        globals={
            "fns": fns
        },
        label=label,
        sub_label=sub_label,
        description=description,
    ).blocked_autorange(min_run_time=min_run_time)

    if NVTX_PROFILE:
        with nvtx.annotate("mm-bench"):
            with nvtx.annotate(f"{label}|{sub_label}|{description}"):
                fns[0]()

    return res


def bench(types: TypeConfig,
          group_size: int,
          m: int,
          k: int,
          n: int,
          label: str,
          sub_label: str,
          kernels: List[str],
          sweep_schedules: bool = True) -> List[TMeasurement]:
    benchmark_tensors = create_bench_tensors((m, n, k), types, group_size)
    sub_label += f", L={len(benchmark_tensors)}"

    timers = []
    for kernel_name in kernels:
        kernel = get_kernel_module(kernel_name)
        
        name = kernel_name
        if hasattr(kernel, "name"):
            name = kernel.name(types)

        timers.append(bench_fns(label, sub_label, name,
            [kernel.create_bench_fn(bt, types.output_type) 
             for bt in benchmark_tensors]))

    return timers


# runner
def print_timers(timers: List[TMeasurement]):
    compare = TBenchmark.Compare(timers)
    compare.print()


def run(args, MKNs: Iterable[Tuple[int, int, int]]) -> Iterable[TMeasurement]:
    
    types = TypeConfig(
        act_type=args.act_type,
        weight_type=scalar_types.uint4b8 if args.group_zero_type is None \
            else scalar_types.uint4,
        output_type=args.out_type,
        group_scale_type=args.group_scale_type,
        group_zero_type=args.group_zero_type,
        channel_scale_type=args.channel_scale_type,
        token_scale_type=args.token_scale_type,
    )
    
    results: List[TMeasurement] = []
    for m, k, n in MKNs:
        timers = bench(types,
                       args.group_size,
                       m,
                       k,
                       n,
                       f"{args.act_type}-gemm",
                       f"MKN=({m}x{k}x{n})",
                       kernels=args.kernels.split(","),
                       sweep_schedules=args.sweep_schedules)
        print_timers(timers)
        results.extend(timers)

    return results


# output makers
def make_output(
    data: List[TMeasurement],
    MKNs: Iterable[Tuple[int, int, int]],
    base_description: str,
    timestamp=None,
):

    print(f"== All Results {base_description} ====")
    print_timers(data)

    # pickle all the results
    timestamp = int(time.time()) if timestamp is None else timestamp
    with open(f"{base_description}-{timestamp}.pkl", "wb") as f:
        pkl.dump(data, f)


# argparse runners


def run_square_bench(args):
    dim_sizes = list(
        range(args.dim_start, args.dim_end + 1, args.dim_increment))
    MKNs = list(zip(dim_sizes, dim_sizes, dim_sizes))
    data = run(args, MKNs)

    make_output(data, MKNs, f"square_bench-{args.dtype}")


def run_range_bench(args):
    dim_sizes = list(range(args.dim_start, args.dim_end, args.dim_increment))
    n = len(dim_sizes)
    Ms = [args.m_constant] * n if args.m_constant is not None else dim_sizes
    Ks = [args.k_constant] * n if args.k_constant is not None else dim_sizes
    Ns = [args.n_constant] * n if args.n_constant is not None else dim_sizes
    MKNs = list(zip(Ms, Ks, Ns))
    data = run(args, MKNs)

    make_output(data, MKNs, f"range_bench-{args.act_type}")


def run_shape_bench(args):
    Ms = [int(m) for m in args.ms.split(",")]
    Ks = [args.k] * len(Ms)
    Ns = [args.n] * len(Ms)
    MKNs = list(zip(Ms, Ks, Ns))
    data = run(args, MKNs)

    make_output(data, MKNs, f"shape_bench-{args.act_type}")


def run_model_bench(args):

    print("Benchmarking models:")
    for i, model in enumerate(args.models):
        print(f"[{i}]  {model}")

    def model_shapes(model_name: str, tp_size: int) -> List[Tuple[int, int]]:
        KNs = []
        for KN, tp_split_dim in copy.deepcopy(WEIGHT_SHAPES[model_name]):
            KN[tp_split_dim] = KN[tp_split_dim] // tp_size
            KNs.append(KN)
        return KNs

    model_bench_data = []
    models_tps = list(itertools.product(args.models, args.tp_sizes))
    for model, tp_size in models_tps:
        Ms = args.batch_sizes
        KNs = model_shapes(model, tp_size)
        MKNs = []
        for m in Ms:
            for k, n in KNs:
                MKNs.append((m, k, n))

        data = run(args, MKNs)
        model_bench_data.append(data)

    type_string = f"{args.act_type}"

    # Print all results
    for data, model_tp in zip(model_bench_data, models_tps):
        model, tp_size = model_tp
        print(f"== Results {type_string} {model}-TP{tp_size} ====")
        print_timers(data)

    timestr = time.strftime("%Y%m%d-%H%M%S")

    all_results = []
    for d in model_bench_data:
        all_results.extend(d)

    # pickle all data
    with open(f"model_bench-{type_string}-{timestr}.pkl", "wb") as f:
        args_dict = vars(args)
        args_dict.pop("func")
        pkl.dump({
            "args": args_dict,
            "results": all_results,
        }, f)


if __name__ == "__main__":

    def to_torch_dtype(dt):
        return {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "int8": torch.int8,
            "float8_e4m3fn": torch.float8_e4m3fn,
            "int": torch.int,
            "float": torch.float,
        }[dt]

    parser = FlexibleArgumentParser(
        description="""
Benchmark Machete GEMM.

    To run square GEMMs:
        python3 ./benchmarks/kernels/benchmark_machete.py --dtype float16 square_bench --dim-start 128 --dim-end 512 --dim-increment 64
    
    To run constant N and K and sweep M:
        python3 ./benchmarks/kernels/benchmark_machete.py --dtype float16 range_bench --dim-start 128 --dim-end 512 --dim-increment 64 --n-constant 16384 --k-constant 16384
    
    To run dimensions from a model:
        python3 ./benchmarks/kernels/benchmark_machete.py --dtype float16 model_bench --models meta-llama/Llama-2-7b-hf --batch-sizes 16 --tp-sizes 1
    
    Output:
        - a .pkl file, that is a list of raw torch.benchmark.utils.Measurements for the pytorch and cutlass implementations for the various GEMMs.
            """,  # noqa: E501
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--act-type",
        type=to_torch_dtype,
        required=True,
        help="Available options are "
        "['bfloat16', 'float16', 'int8', 'float8_e4m3fn']",
    )
    parser.add_argument(
        "--group-scale-type",
        type=to_torch_dtype,
        help="Available options are ['bfloat16', 'float16']",
    )
    parser.add_argument(
        "--group-zero-type",
        type=to_torch_dtype,
        help="Available options are ['bfloat16', 'float16']",
    )
    parser.add_argument(
        "--channel-scale-type",
        type=to_torch_dtype,
        help="Available options are ['bfloat16', 'float16', 'float']",
    )
    parser.add_argument(
        "--token-scale-type",
        type=to_torch_dtype,
        help="Available options are ['bfloat16', 'float16', 'float']",
    )
    parser.add_argument(
        "--out-type",
        type=to_torch_dtype,
        help="Available options are "
        "['bfloat16', 'float16', 'int', 'float']",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        help="Available options are ['None', '-1', '128'], default=128",
        default=128,
    )
    parser.add_argument(
        "--sweep-schedules",
        action="store_true",
        help="Run a sweep over all supported schedules",
    )
    parser.add_argument(
        "--kernels",
        type=str,
        required=True,
    )
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    square_parser = subparsers.add_parser("square_bench")
    square_parser.add_argument("--dim-start", type=int, required=True)
    square_parser.add_argument("--dim-end", type=int, required=True)
    square_parser.add_argument("--dim-increment", type=int, required=True)
    square_parser.set_defaults(func=run_square_bench)

    shape_parser = subparsers.add_parser("shape_bench")
    shape_parser.add_argument("--ms", type=str, default=None)
    shape_parser.add_argument("--n", type=int, default=None)
    shape_parser.add_argument("--k", type=int, default=None)
    shape_parser.set_defaults(func=run_shape_bench)
    
    range_parser = subparsers.add_parser("range_bench")
    range_parser.add_argument("--dim-start", type=int, required=True)
    range_parser.add_argument("--dim-end", type=int, required=True)
    range_parser.add_argument("--dim-increment", type=int, required=True)
    range_parser.add_argument("--m-constant", type=int, default=None)
    range_parser.add_argument("--n-constant", type=int, default=None)
    range_parser.add_argument("--k-constant", type=int, default=None)
    range_parser.set_defaults(func=run_range_bench)

    model_parser = subparsers.add_parser("model_bench")
    model_parser.add_argument(
        "--models",
        nargs="+",
        type=str,
        default=DEFAULT_MODELS,
        choices=WEIGHT_SHAPES.keys(),
    )
    model_parser.add_argument("--tp-sizes",
                              nargs="+",
                              type=int,
                              default=DEFAULT_TP_SIZES)
    model_parser.add_argument("--batch-sizes",
                              nargs="+",
                              type=int,
                              default=DEFAULT_BATCH_SIZES)
    model_parser.set_defaults(func=run_model_bench)

    args = parser.parse_args()
    args.func(args)
