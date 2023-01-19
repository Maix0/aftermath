{
  description = "A basic flake with a shell";
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
  inputs.flake-utils.url = "github:numtide/flake-utils";
  inputs.rust-overlay.url = "github:oxalica/rust-overlay";
  inputs.cargo-workspace.url = "github:Maix0/cargo-ws-flake";
  inputs.cargo-semver-checks.url = "github:Maix0/cargo-semver-checks-flake";
  outputs = {
    self,
    nixpkgs,
    flake-utils,
    rust-overlay,
    cargo-workspace,
    cargo-semver-checks,
  }:
    flake-utils.lib.eachDefaultSystem (system: let
      pkgs = import nixpkgs {
        inherit system;
        overlays = [(import rust-overlay)];
        config.android_sdk.accept_license = true;
      };
    in {
      devShell = with pkgs; let
        cargo-ws = cargo-workspace.packages.${system}.default;
        cargo-sc = cargo-semver-checks.packages.${system}.default;
        rust_dev = rust-bin.selectLatestNightlyWith (toolchain:
          toolchain.default.override {
            extensions = ["rust-src"];

            targets = ["wasm32-unknown-unknown" "x86_64-unknown-linux-gnu"];
          });
      in
        mkShell {
          nativeBuildInputs = [
            pkgs.bashInteractive
          ];
          buildInputs = [
            # Rust
            cmake
            rust_dev
            cargo-sc
            #cargo-semver-checks
            cargo-ws
          ];
        };
    });
}
