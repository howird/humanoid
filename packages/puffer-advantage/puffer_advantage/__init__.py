import torch
import torch.utils.cpp_extension

from pathlib import Path

# TODO(howird): dynamically compile/load for now https://pytorch.org/tutorials/advanced/cpp_custom_ops.html#cpp-custom-ops-tutorial
ROOT_DIR = Path(__path__[0])
if torch.cuda.is_available():
    ext = "cu"
else:
    ext = "cpp"
    print("WARNING: CUDA is not available.")

src = ROOT_DIR / f"pufferlib.{ext}"

puffer_advantage = torch.utils.cpp_extension.load(name="puffer_advantage", sources=[str(src)], verbose=True)
# compute_gae = puffer_cuda.compute_gae  # type: ignore
# compute_vtrace = puffer_cuda.compute_vtrace  # type: ignore
# compute_puff_advantage = puffer_cuda.compute_puff_advantage  # type: ignore
