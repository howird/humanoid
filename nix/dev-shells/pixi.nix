{ lib
, stdenv
, mkShell
, xorg
, linuxPackages
, gcc
, pixi
, libxcrypt
}:
mkShell rec {
  name = "humanoid";

  packages = [
    cudatoolkit
    linuxPackages.nvidia_x11

    gcc
    pixi
  ];

  PROJ_DIR = builtins.toString ./../..;
  ENV_DIR = "${PROJ_DIR}/.pixi/envs/default";

  LIBRARIES = [
    stdenv.cc.cc
    xorg.libX11
    linuxPackages.nvidia_x11
    "${ENV_DIR}"
    "${ENV_DIR}/lib/python3.8/site-packages/torch"
  ];

  INCLUDES = [
    libxcrypt
    "${ENV_DIR}"
    "${ENV_DIR}/lib/python3.8/site-packages/torch"
    "${ENV_DIR}/lib/python3.8/site-packages/torch/include/torch/csrc/api"
  ];

  NIX_LD = builtins.readFile "${stdenv.cc}/nix-support/dynamic-linker";
  NIX_LD_LIBRARY_PATH = lib.makeLibraryPath LIBRARIES;
  CPATH = lib.makeIncludePath INCLUDES + ":${ENV_DIR}/include/python3.8";

  shellHook = ''
    export LD_LIBRARY_PATH=$NIX_LD_LIBRARY_PATH
  '';
}
