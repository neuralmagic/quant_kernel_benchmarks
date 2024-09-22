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

    args = parser.parse_args()

    with open(args.filename, 'rb') as f:
        data: List[TMeasurement] = pickle.load(f)


    results = defaultdict(lambda: list())
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

        # Remove the BASELINE method from the plot
        plot_df = normalized_df[normalized_df['kernel'] != BASELINE]

        # Create a custom palette that ensures the highlighted kernel is the most red
        kernels = plot_df['kernel'].unique()
        kernel_colors = {kernel: color for kernel, color in zip(kernels, color_palette)}
    
        # Swap colors to make the highlighted kernel the most red
        if args.highlight in kernel_colors:
            highlight_color = kernel_colors[args.highlight]
            kernel_colors[args.highlight] = color_palette[most_red_index]
            kernel_colors[kernels[most_red_index]] = highlight_color

        sns.lineplot(data=plot_df,
                     x="batch_size",
                     y="normalized_runtime",
                     hue="kernel",
                     markers=True,
                     dashes=False,
                     palette=kernel_colors,
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
    outfile = "graph_bench_normalized_runtime"
    
    plt.savefig(f"{outfile}.pdf", bbox_inches='tight')
    print("Saved to", f"{outfile}.pdf")