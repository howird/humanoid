{ pkgs ? import <nixpkgs> {} }:
(pkgs.buildFHSUserEnv {
  name = "cuda";
  targetPkgs = pkgs: with pkgs; [ 
    # git gitRepo
    # gnupg
    # autoconf
    # curl
    # procps
    # gnumake
    # util-linux
    # m4
    # gperf
    # unzip
    # zlib 

    # cudatoolkit linuxPackages.nvidia_x11
    # libGLU libGL
    # xorg.libXi xorg.libXmu freeglut
    # xorg.libXext xorg.libX11 xorg.libXv xorg.libXrandr
    # ncurses5
    # stdenv.cc
    # binutils

    # pixi

    # cmake
    # mesa
    # vulkan-loader
    # vulkan-headers
    # vulkan-tools
  ];
  # runScript = "${pkgs.bash}/bin/bash";
  runScript = "${pkgs.zsh}/bin/zsh";
  # profile = ''
  #   export CUDA_PATH=${pkgs.cudatoolkit}
  #   # export LD_LIBRARY_PATH=${pkgs.linuxPackages.nvidia_x11}/lib
  #   export EXTRA_LDFLAGS="-L/lib -L${pkgs.linuxPackages.nvidia_x11}/lib"
  #   export EXTRA_CCFLAGS="-I/usr/include"
  # '';
}).env