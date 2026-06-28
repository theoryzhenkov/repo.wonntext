{
  description = "WONNText development shell";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };
      in
      {
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            age
            git
            jujutsu
            sops
            stdenv.cc.cc.lib
            uv
            python313
          ];

          shellHook = ''
            export UV_PYTHON_DOWNLOADS=never
            export UV_PYTHON="${pkgs.python313}/bin/python"
            export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
            export PATH="$PWD/.venv/bin:''${PATH}"
          '';
        };
      });
}
