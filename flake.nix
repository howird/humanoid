{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    utils.url = "github:numtide/flake-utils";
    ml-dev = {
      url = "github:howird/ml-dev-flake";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = {
    nixpkgs,
    ml-dev,
    ...
  } @ inputs:
    inputs.utils.lib.eachSystem ["x86_64-linux"] (system: {
      devShells = {
        default = ml-dev.devShells.${system}.default;
      };
      formatter = nixpkgs.legacyPackages.${system}.alejandra;
    });
}
