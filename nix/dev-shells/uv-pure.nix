{ lib
, pkgs
, pyproject-nix
, pyproject-build-systems
, uv2nix
, mkShell
, python310
}:
let
  workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ./.; };
  overlay = workspace.mkPyprojectOverlay {
    sourcePreference = "wheel"; # or sourcePreference = "sdist";
  };

  pyprojectOverrides = _final: _prev: {
    # Implement build fixups here.
    # Note that uv2nix is _not_ using Nixpkgs buildPythonPackage.
    # It's using https://pyproject-nix.github.io/pyproject.nix/build.html
  };

  python = python310;

  pythonSet =
    # Use base package set from pyproject.nix builders
    (pkgs.callPackage pyproject-nix.build.packages {
      inherit python;
    }).overrideScope
      (
        lib.composeManyExtensions [
          pyproject-build-systems.overlays.default
          overlay
          pyprojectOverrides
        ]
      );

  # Create an overlay enabling editable mode for all local dependencies.
  editableOverlay = workspace.mkEditablePyprojectOverlay {
    # Use environment variable
    root = "$REPO_ROOT";
    # Optional: Only enable editable for these packages
    # members = [ "hello-world" ];
  };

  # Override previous set with our overrideable overlay.
  editablePythonSet = pythonSet.overrideScope (
    lib.composeManyExtensions [
      editableOverlay

      # Apply fixups for building an editable package of your workspace packages
      (final: prev: {
        hello-world = prev.hello-world.overrideAttrs (old: {
          # It's a good idea to filter the sources going into an editable build
          # so the editable package doesn't have to be rebuilt on every change.
          src = lib.fileset.toSource {
            root = old.src;
            fileset = lib.fileset.unions [
              (old.src + "/pyproject.toml")
              (old.src + "/README.md")
              (old.src + "/src/hello_world/__init__.py")
            ];
          };

          # Hatchling (our build system) has a dependency on the `editables` package when building editables.
          #
          # In normal Python flows this dependency is dynamically handled, and doesn't need to be explicitly declared.
          # This behaviour is documented in PEP-660.
          #
          # With Nix the dependency needs to be explicitly declared.
          nativeBuildInputs =
            old.nativeBuildInputs
            ++ final.resolveBuildSystem {
              editables = [ ];
            };
        });

      })
    ]
  );

  # Build virtual environment, with local packages being editable.
  #
  # Enable all optional dependencies for development.
  virtualenv = editablePythonSet.mkVirtualEnv "hello-world-dev-env" workspace.deps.all;

in
mkShell {
  packages = [
    virtualenv
    pkgs.uv
  ];

  env = {
    # Don't create venv using uv
    UV_NO_SYNC = "1";

    # Force uv to use Python interpreter from venv
    UV_PYTHON = "${virtualenv}/bin/python";

    # Prevent uv from downloading managed Python's
    UV_PYTHON_DOWNLOADS = "never";
  };

  shellHook = ''
    # Undo dependency propagation by nixpkgs.
    unset PYTHONPATH

    # Get repository root using git. This is expanded at runtime by the editable `.pth` machinery.
    export REPO_ROOT=$(git rev-parse --show-toplevel)
  '';
}
