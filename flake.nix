{
  description = "pythonise your mafs";
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-25.05";
  };
  outputs = { nixpkgs, ... }: {
    devShells.x86_64-linux.default = let
      pkgs = nixpkgs.legacyPackages.x86_64-linux;
    in pkgs.mkShellNoCC {
      packages = with pkgs; [
        python313
        python313Packages.pandas
        python313Packages.numpy
        python313Packages.pip
        python313Packages.sklearn-compat
        python313Packages.matplotlib
      ];
      pythonPath = with pkgs; [
          python313Packages.pandas
          python313Packages.numpy
          python313Packages.pip
          python313Packages.sklearn-compat
          python313Packages.matplotlib
      ];
      # if you're competent, you'll add the script for
      ## python -m venv ex
      ## source ex/bin/activate
      ## pip install factor_analyzer
      ## python3 stats.py
      # when running the app from the flake
      # i was paid to do the python, not to do the nix shell wonk
    };
  };
}
