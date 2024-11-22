{
  description = "An awesome machine-learning project";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.05";
    utils.url = "github:numtide/flake-utils";

    # ml-pkgs.url = "github:howird/ml-pkgs/howird/maniskill3";
    ml-pkgs.url = "github:howird/ml-pkgs/626a57a03409bd56d1d0b74a95556dbc5b510acb";
    ml-pkgs.inputs.nixpkgs.follows = "nixpkgs";
    ml-pkgs.inputs.utils.follows = "utils";

    # nixpkgs-python.url = "github:cachix/nixpkgs-python";
  };

  outputs = {
    self,
    nixpkgs,
    ...
  } @ inputs:
    {
      overlays.default = nixpkgs.lib.composeManyExtensions [
        inputs.ml-pkgs.overlays.default
        # (final: prev: {
        #   pythonPackagesExtensions = prev.pythonPackagesExtensions ++ [
        #     (python-final: python-prev: {
        #       my-package = ...;
        #     })
        #   ];
        # })
      ];
    } // inputs.utils.lib.eachSystem [ "x86_64-linux" ] (system:
    let
      pkgs = import nixpkgs {
        inherit system;
        config = {
          allowUnfree = true;
          cudaSupport = true;
          cudaCapabilities = [ "7.5" "8.6" ];
          cudaForwardCompat = true;
        };
        overlays = [
          self.overlays.default
        ];
      };
    in {
      devShells = {
        default = pkgs.callPackage ./dev-shells/default.nix {};
        pixi = pkgs.callPackage ./dev-shells/pixi.nix {};
        mujoco = pkgs.callPackage ./dev-shells/mujoco.nix {};
      };
    });
}
