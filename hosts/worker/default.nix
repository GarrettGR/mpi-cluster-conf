{config, pkgs, ...}: {
  imports = [
    ../../modules/nixos/common.nix
    ../../modules/nixos/nvidia.nix
    ../../modules/nixos/networking.nix
  ];

  boot = {
    supportedFilesystems = [ "nfs" "nfs4" ];
    initrd.kernelModules = [ "nfs" ];
  };

  cluster.networking = {
    hostName = "node1";
    ipAddress = "10.0.0.2";
    publicAddress = "192.168.1.202";
    interface = "enp0s20f0u2";
  };

  services.rpcbind.enable = true;

  fileSystems = {
    "/common/project" = {
      device = "10.0.0.1:/project";
      fsType = "nfs";
      options = [
        "nofail"
        "sync"
        "rsize=1048576"
        "wsize=1048576"
        "nfsvers=4.2"
      ];
    };
    "/common/scratch" = {
      device = "10.0.0.1:/scratch";
      fsType = "nfs";
      options = [
        "nofail"
        "sync"
        "rsize=1048576"
        "wsize=1048576"
        "nfsvers=4.2"
      ];
    };
  };
}
