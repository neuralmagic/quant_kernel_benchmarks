import importlib

def get_kernel_module(kernel_name):
    return importlib.import_module(f"kernels.{kernel_name}")