{
  description = "Generals.io bot framework — JAX-based simulator for RL research";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };

        # Native libraries needed by pip-installed wheels (jaxlib, pygame, scipy)
        nativeLibs = with pkgs; [
          stdenv.cc.cc.lib   # libstdc++.so.6
          zlib               # libz.so.1
          glib               # libglib-2.0.so, libgthread-2.0.so
          SDL2
          SDL2_image
          SDL2_mixer
          SDL2_ttf
        ];

        python = pkgs.python311;

        shellHook = ''
          export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath nativeLibs}:$LD_LIBRARY_PATH"

          # Create venv + install deps if needed
          if [ ! -d .venv ]; then
            uv sync
          fi
          export PATH="$PWD/.venv/bin:$PATH"
        '';
      in
      {
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            python
            uv
            # For cage-based GUI display on NixOS servers
            cage
          ];

          inherit shellHook;
        };

        # nix run .#server -- --grid 15
        apps.server = {
          type = "app";
          program = toString (pkgs.writeShellScript "lan-server" ''
            export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath nativeLibs}:$LD_LIBRARY_PATH"
            cd "${self}" 2>/dev/null || true
            if [ -d .venv ]; then
              exec .venv/bin/python3 examples/lan_server.py "$@"
            else
              exec ${pkgs.uv}/bin/uv run python3 examples/lan_server.py "$@"
            fi
          '');
        };

        # nix run .#client -- --host 192.168.1.10
        apps.client = {
          type = "app";
          program = toString (pkgs.writeShellScript "lan-client" ''
            export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath nativeLibs}:$LD_LIBRARY_PATH"
            cd "${self}" 2>/dev/null || true
            if [ -d .venv ]; then
              exec .venv/bin/python3 examples/lan_client.py "$@"
            else
              exec ${pkgs.uv}/bin/uv run python3 examples/lan_client.py "$@"
            fi
          '');
        };
      });
}
