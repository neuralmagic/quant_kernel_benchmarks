import math
import pickle
import re
from collections import defaultdict
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from torch.utils.benchmark import Measurement as TMeasurement

from vllm.utils import FlexibleArgumentParser

BASELINE = "torch.mm (fp16)"

# Set the font to Roboto
plt.rcParams['font.family'] = 'Roboto'

if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description='Benchmark the latency of processing a single batch of '
        'requests till completion.')
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
    for v in data["results"]:
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

        kernel = v.task_spec.description
        all_kernels.add(kernel)
        if kernel not in args.ignore:  # Only add results for kernels not in the ignore list
            results[KN].append({
                "kernel": kernel,
                "batch_size": M,
                "median": v.median
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

    # Define a custom color palette
    color_palette = sns.color_palette("husl", 8)
    # Find the most red color in the palette
    most_red_index = max(range(len(color_palette)), key=lambda i: color_palette[i][0])

    # Assign colors to all kernels, including ignored ones
    all_kernels_list = sorted(list(all_kernels))
    kernel_colors = {kernel: color for kernel, color in zip(all_kernels_list, color_palette)}

    # Ensure the highlighted kernel is the most red, even if it's ignored
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

        def normalize_group(group):
            torch_mm_runtime = group[group['kernel'] == BASELINE]['median'].values[0]
            group['normalized_runtime'] = group['median'] / torch_mm_runtime
            return group

        # Group by batch_size, apply normalization, and reset index
        normalized_df = df.groupby('batch_size')\
            [['batch_size', 'kernel', 'median']]\
            .apply(normalize_group).reset_index(drop=True)

        # Reorder columns for better readability
        def _key(x):
            if x.name == "kernel":
                # make sure the highlighted kernel is the last one so its on top
                x[x == args.highlight] = 'zzzzzzzz'
            return x
            
        normalized_df = normalized_df[['kernel', 'batch_size', 'median', 'normalized_runtime']]
        normalized_df = normalized_df.sort_values(by=['batch_size', 'kernel'], key=_key)

        normalized_df['batch_size'] = normalized_df['batch_size'].astype(str)

        # Remove the BASELINE method and ignored kernels from the plot
        plot_df = normalized_df[(normalized_df['kernel'] != BASELINE) & (~normalized_df['kernel'].isin(args.ignore))]

        # Use the pre-assigned colors for the plot
        plot_colors = {k: v for k, v in kernel_colors.items() if k in plot_df['kernel'].unique()}

        sns.lineplot(data=plot_df,
                     x="batch_size",
                     y="normalized_runtime",
                     hue="kernel",
                     markers=True,
                     dashes=False,
                     palette=plot_colors,
                     marker="o")  

        plt.title(f"Weight Shape: {shape} (OUTxIN)", fontsize=14)
        plt.ylim(0, 3.1)
        plt.ylabel("Normalized Latency\n(FP16 torch.mm)", fontsize=12)
        plt.xlabel("Batch Size / Seq. Len", fontsize=12)

        # Add grey dotted line at y=1
        plt.axhline(y=1, color='grey', linestyle=':', linewidth=1)

        # Customize the legend
        legend = plt.legend(title="Kernel", title_fontsize='12', fontsize='10', loc='upper left', bbox_to_anchor=(1, 1))
        
        # Update legend markers to circles
        for handle in legend.legend_handles:
            handle.set_marker('o')

        # Adjust tick label font size
        plt.tick_params(axis='both', which='major', labelsize=10)

    # Remove any unused subplots
    for i in range(axs_idx + 1, len(axs)):
        fig.delaxes(axs[i])

    plt.tight_layout()
    plt.gca().spines[['right', 'top']].set_visible(False)
    outfile = "graph_bench_normalized_runtime" if args.outfile is None else args.outfile
    plt.savefig(f"{outfile}.svg", bbox_inches='tight', transparent=True)
    print("Saved to", f"{outfile}.svg")