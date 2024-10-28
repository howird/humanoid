import sys
import torch


def make_tensor(device="cuda"):
    return torch.rand((10, 10), device=device)


def check_cuda():
    make_tensor()
    return torch.cuda.is_available() and (
        torch.version.cuda is not None
    )


def check_rocm():
    make_tensor()
    return torch.cuda.is_available() and (
        torch.version.hip is not None
    )


def check_cpu():
    make_tensor("cpu")
    return not torch.cuda.is_available()


if __name__ == "__main__":
    print(f"Hello from {__file__}! Args: {sys.argv[1:] if len(sys.argv) > 1 else 'None'}")
    print(f"")

    if check_cuda():
        print("CUDA is available")
    elif check_rocm():
        print("ROCm is available")
    elif check_cpu():
        print("CPU is available")
    else:
        raise ValueError("CUDA, ROCm, CPU all not available")
