{pkgs ? import <nixpkgs> {}}:
(pkgs.buildFHSEnv rec {
  name = "skillmimic";

  targetPkgs = pkgs: with pkgs; [
    gcc

    linuxPackages.nvidia_x11
    cudatoolkit

    libxcrypt

    pixi
  ];

  PROJ_DIR = builtins.toString ./.;
  ENV_DIR = "${PROJ_DIR}/.pixi/envs/${name}";

  LIBRARIES = with pkgs; [
    # glibc
    # stdenv.cc.cc
    # linuxPackages.nvidia_x11
    "${ENV_DIR}"
    "${ENV_DIR}/lib/python3.8/site-packages/torch"
  ];

  INCLUDES = with pkgs; [
    libxcrypt
  #   "${ENV_DIR}"
  #   "${ENV_DIR}/lib/python3.8/site-packages/torch"
  #   "${ENV_DIR}/lib/python3.8/site-packages/torch/include/torch/csrc/api"
  ];

  runScript = "zsh";
  profile = ''
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${pkgs.lib.makeLibraryPath LIBRARIES}
    export CPATH=${pkgs.lib.makeIncludePath INCLUDES}:${ENV_DIR}/include/python3.8
  '';
}).env
