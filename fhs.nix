{pkgs ? import <nixpkgs> {}}:
(pkgs.buildFHSUserEnv rec {
  name = "skillmimic";

  targetPkgs = pkgs: with pkgs; [
    gcc
    stdenv.cc.cc

    linuxPackages.nvidia_x11
    cudatoolkit

    libxcrypt

    pixi
  ];

  PROJ_DIR = builtins.toString ./.;
  ENV_DIR = "${PROJ_DIR}/.pixi/envs/${name}";

  profile = ''
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${ENV_DIR}/lib
  '';
}).env
