# Example Usage

Run the benchmark (generates a .pkl file with the results)

```
python benchmark_kernels.py --act-type bfloat16 --kernels torch_fp16,machete,fbgemm_i4,marlin,gemlite model_bench
```

Plot the results

```
python plot/plot_normalized_runtime.py <generated_file>.pkl --highlight machete
```