import os
import sys

import torch


def make_tensor(device="cuda"):
    return torch.rand((10, 10), device=device)


def check_cuda():
    assert torch.cuda.is_available() and (
        torch.version.cuda is not None
    ), "CUDA is not available"
    print(make_tensor())
    print("CUDA is available, in cuda environment as expected")


def check_cpu():
    assert not torch.cuda.is_available(), "CUDA is available, in default environment"
    print(make_tensor("cpu"))
    print("CUDA is not available, in default environment as expected")


if __name__ == "__main__":
    print(f"Hello from {__file__}!")
    print(f"Args: {sys.argv[1:] if len(sys.argv) > 1 else 'None'}")

    if "PIXI_ENVIRONMENT_NAME" in os.environ:
        print(f'Running on pixi env: {os.environ["PIXI_ENVIRONMENT_NAME"]}')

    if sys.argv[1] == "--cuda":
        check_cuda()

    if sys.argv[1] == "--cpu":
        check_cpu()
