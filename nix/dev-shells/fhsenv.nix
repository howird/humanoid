{ lib
, pkgs
, buildFHSEnv
, config
, python310
}:
(buildFHSEnv {
  name = "uv";

  targetPkgs = pkgs: (with pkgs; [
    pythonManylinuxPackages.manylinux1
    libGL glfw glm # mujoco
  ]) ++ (lib.lists.optional config.cudaSupport (with pkgs.cudaPackages; [
    cuda_cudart
    # cuda_nvcc
    cuda_cupti

    cudnn
    nccl
    libcusparse
    libcurand
    libcusolver
    libcufft
    libcublas

    pkgs.linuxPackages.nvidia_x11_vulkan_beta
  ]));

  multiPkgs = pkgs: (with pkgs; [
  ]);

  # see https://nixos.org/manual/nixpkgs/stable/#how-to-consume-python-modules-using-pip-in-a-virtual-environment-like-i-am-used-to-on-other-operating-systems
  # export CUDA_PATH=${pkgs.cudatoolkit}
  profile = ''
    export UV_PYTHON_DOWNLOADS=never
    export UV_PYTHON=${python310.interpreter};

    SOURCE_DATE_EPOCH=$(date +%s)
    if [ -d ".venv" ]; then
        echo "Skipping venv creation, '.venv' already exists"
    else
        echo "Creating new venv environment in path: '.venv'"
        uv venv
    fi
    source ".venv/bin/activate"
    uv sync --extra ${if config.cudaSupport then "cuda" else "cpu"}
  '';

  runScript = "bash";
}).env
