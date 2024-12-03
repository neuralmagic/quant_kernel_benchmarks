import math
import pickle
import re
from collections import defaultdict
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from torch.utils.benchmark import Measurement as TMeasurement

from vllm.utils import FlexibleArgumentParser

def calculate_flops(M: int, K: int, N: int, L: int) -> float:
    return 2 * M * N * K * L # GEMM FLOPS

# Set the font to Roboto
plt.rcParams['font.family'] = 'Roboto'

if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description='Benchmark the percentage of peak FLOPS for processing a single batch of requests.')
    parser.add_argument('filename', type=str)
    parser.add_argument('--shape', type=str, help='Specific shape to plot (e.g., "1024x1024")')
    parser.add_argument('--highlight', type=str, default='machete', help='Kernel to highlight')
    parser.add_argument('--ignore', type=str, nargs='+', default=[], help='Kernels to ignore (not plot)')
    parser.add_argument('--outfile', type=str, help='Output file name')
    args = parser.parse_args()

    with open(args.filename, 'rb') as f:
        data: List[TMeasurement] = pickle.load(f)

    results = defaultdict(lambda: list())
    all_kernels = set()

    if "results" in data:
        data = data["results"]
    
    for v in data:
        print(v.task_spec.sub_label)
        L = 1
        result = re.search(r"L=(\d+)", v.task_spec.sub_label)
        if result is not None:
            L = int(result.group(1))

        print(L)

        result = re.search(r"MKN=\(\d+x(\d+x\d+)\)", v.task_spec.sub_label)
        if result is not None:
            KN = result.group(1)
        else:
            raise Exception("MKN not found")
        result = re.search(r"MKN=\((\d+)x(\d+)x(\d+)\)", v.task_spec.sub_label)
        if result is not None:
            M, K, N = map(int, result.groups())
        else:
            raise Exception("MKN not found")
        
        print(M, K, N, L)

        kernel = v.task_spec.description
        all_kernels.add(kernel)
        if kernel not in args.ignore:
            flops = calculate_flops(M, K, N, L)
            tlops = (flops / v.median) / 1e12
            results[KN].append({
                "kernel": kernel,
                "batch_size": M,
                "median": v.median,
                "tlops": tlops
            })

    if args.shape:
        if args.shape not in results:
            print(f"Shape {args.shape} not found in the data.")
            exit(1)
        shapes_to_plot = [args.shape]
        rows, cols = 1, 1
    else:
        shapes_to_plot = list(results.keys())
        rows = int(math.ceil(len(shapes_to_plot) / 2))
        cols = 2

    fig, axs = plt.subplots(rows, cols, figsize=(7 * cols, 4 * rows))
    if not isinstance(axs, np.ndarray):
        axs = np.array([axs])
    axs = axs.flatten()

    color_palette = sns.color_palette("husl", 8)
    most_red_index = max(range(len(color_palette)), key=lambda i: color_palette[i][0])

    all_kernels_list = sorted(list(all_kernels))
    kernel_colors = {kernel: color for kernel, color in zip(all_kernels_list, color_palette)}

    if args.highlight in kernel_colors:
        highlight_color = kernel_colors[args.highlight]
        kernel_colors[args.highlight] = color_palette[most_red_index]
        kernel_colors[all_kernels_list[most_red_index]] = highlight_color

    kernel_colors = {
        "torch.mm (fp16)": "#FFC93F",
        "machete": "#2A8EFD",
        "marlin": "#03C883",
        "gemlite": "#7E7F86",
        "fbgemm_i4": "#D3D4DD"
    }

    for axs_idx, shape in enumerate(shapes_to_plot):
        plt.sca(axs[axs_idx])
        df = pd.DataFrame(results[shape])

        df = df.sort_values(by=['batch_size', 'kernel'])
        df['batch_size'] = df['batch_size'].astype(str)

        plot_df = df[~df['kernel'].isin(args.ignore)]

        plot_colors = {k: v for k, v in kernel_colors.items() if k in plot_df['kernel'].unique()}

        sns.lineplot(data=plot_df,
                     x="batch_size",
                     y="tlops",
                     hue="kernel",
                     markers=True,
                     dashes=False,
                     palette=plot_colors,
                     marker="o")  

        plt.title(f"Weight Shape: {shape} (OUTxIN)", fontsize=14)
        plt.ylabel("TFLOPS/s", fontsize=12)
        plt.xlabel("Batch Size / Seq. Len", fontsize=12)

        legend = plt.legend(title="Kernel", title_fontsize='12', fontsize='10', loc='upper left', bbox_to_anchor=(1, 1))
        
        for handle in legend.legend_handles:
            handle.set_marker('o')

        plt.tick_params(axis='both', which='major', labelsize=10)

    for i in range(axs_idx + 1, len(axs)):
        fig.delaxes(axs[i])

    plt.tight_layout()
    outfile = f"graph_bench_tflops" if args.outfile is None else args.outfile
    
    plt.gca().spines[['right', 'top']].set_visible(False)
    plt.savefig(f"{outfile}.svg", bbox_inches='tight', transparent=True)
    print(f"Saved to {outfile}.svg")