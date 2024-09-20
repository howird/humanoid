{pkgs ? import <nixpkgs> {}}:
pkgs.mkShell rec {
  name = "skillmimic";
  packages = with pkgs; [
    cudatoolkit
    linuxPackages.nvidia_x11

    gcc
    pixi
  ];

  PROJ_DIR = builtins.toString ./.;
  ENV_DIR = "${PROJ_DIR}/.pixi/envs/${name}";

  LIBRARIES = with pkgs; [
    stdenv.cc.cc
    linuxPackages.nvidia_x11
    "${ENV_DIR}"
    "${ENV_DIR}/lib/python3.8/site-packages/torch"
  ];

  INCLUDES = with pkgs; [
    libxcrypt
    "${ENV_DIR}"
    "${ENV_DIR}/lib/python3.8/site-packages/torch"
    "${ENV_DIR}/lib/python3.8/site-packages/torch/include/torch/csrc/api"
  ];

  NIX_LD = builtins.readFile "${pkgs.stdenv.cc}/nix-support/dynamic-linker";
  NIX_LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath LIBRARIES;
  CPATH = pkgs.lib.makeIncludePath INCLUDES + ":${ENV_DIR}/include/python3.8";

  shellHook = ''
    export LD_LIBRARY_PATH=$NIX_LD_LIBRARY_PATH
    ${pkgs.pixi}/bin/pixi shell -e ${name}
  '';
}
