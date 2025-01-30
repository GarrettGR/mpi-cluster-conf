{ config, pkgs, ... }:

{
  home.username = "mpiuser";
  home.homeDirectory = "/home/mpiuser";
  home.stateVersion = "24.11";

  programs.home-manager.enable = true;

  programs.ssh = {
    enable = true;
    
    matchBlocks = {
      "node0" = {
        hostname = "10.0.0.1";
        user = "mpiuser";
        identityFile = "~/.ssh/mpi_cluster";
      };
      "node1" = {
        hostname = "10.0.0.2";
        user = "mpiuser";
        identityFile = "~/.ssh/mpi_cluster";
      };
    };
  };

  home.activation = {
    generateSshKey = lib.hm.dag.entryAfter ["writeBoundary"] ''
      if [ ! -f "${config.home.homeDirectory}/.ssh/mpi_cluster" ]; then
        $DRY_RUN_CMD mkdir -p "${config.home.homeDirectory}/.ssh"
        $DRY_RUN_CMD ssh-keygen -t ed25519 -N "" -f "${config.home.homeDirectory}/.ssh/mpi_cluster" -C "mpiuser@cluster"
      fi
    '';
  };
}
