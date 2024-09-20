import os
import sys

import torch

def make_tensor(device="cuda"):
    return torch.rand((10, 10), device=device)

if os.environ["PIXI_ENVIRONMENT_NAME"] == "rocm":
    assert torch.cuda.is_available() and (torch.version.hip is not None), "HIP is not available"
    print(make_tensor())
    print("ROCm is available, in rocm environment as expected")

if os.environ["PIXI_ENVIRONMENT_NAME"] == "cuda":
    assert torch.cuda.is_available() and (torch.version.cuda is not None), "CUDA is not available"
    print(make_tensor())
    print("CUDA is available, in cuda environment as expected")

if os.environ["PIXI_ENVIRONMENT_NAME"] == "default":
    assert not torch.cuda.is_available(), "CUDA is available, in default environment"
    print(make_tensor())
    print("CUDA is not available, in default environment as expected")

print("\nHello from train.py!")
print("Environment you are running on:")
print(os.environ["PIXI_ENVIRONMENT_NAME"])
print("Arguments given to the script:")
print(sys.argv[1:])
