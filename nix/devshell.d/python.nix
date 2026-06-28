{ pkgs, theor-project, ... }:
{
  packages =
    theor-project.lib.toolPackages pkgs [
      "uv"
    ]
    ++ [ pkgs.python313 ];

  env = {
    # Use the nix-provided interpreter rather than letting uv download one.
    UV_PYTHON_DOWNLOADS = "never";
    UV_PYTHON = "${pkgs.python313}/bin/python";
  };
}
