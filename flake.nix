{
  description = "NixOS MPI Cluster Configuration";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-24.11";
    
    home-manager = {
      url = "github:nix-community/home-manager/release-24.11";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, home-manager, ... }:
    let
      system = "x86_64-linux";
      pkgs = nixpkgs.legacyPackages.${system};
      
      lib = nixpkgs.lib;

      mkSystem = hostname: nixpkgs.lib.nixosSystem {
        inherit system;
        modules = [
          ./hosts/${hostname}
          home-manager.nixosModules.home-manager
          {
            home-manager.useGlobalPkgs = true;
            home-manager.useUserPackages = true;
            home-manager.users.mpiuser = import ./modules/home-manager/mpiuser.nix;
          }
        ];
      };
    in
    {
      nixosConfigurations = {
        node0 = mkSystem "manager";
        node1 = mkSystem "worker";
      };
    };
}
