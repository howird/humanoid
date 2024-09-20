{
  pkgs ? import <nixpkgs> {},
  lib ? pkgs.lib,
}:
pkgs.mkShell rec {
  name = "isaacgym";
  packages = with pkgs; [
    cudatoolkit
    linuxPackages.nvidia_x11

    pixi
  ];

  LD_LIBRARY_PATH =
    lib.makeLibraryPath (with pkgs; [linuxPackages.nvidia_x11])
    + ":${builtins.toString ./.}/.pixi/envs/${name}/lib";

  shellHook = ''
    pixi shell -e ${name}
  '';
}
