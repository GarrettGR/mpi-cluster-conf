{config, pkgs, ...}: {
  imports = [
    ../../modules/nixos/common.nix
    ../../modules/nixos/nvidia.nix
    ../../modules/nixos/networking.nix
  ];

  cluster.networking = {
    hostName = "node0";
    ipAddress = "10.0.0.1";
    publicAddress = "192.168.1.201";
    interface = "enp0s20f0u2";
  };

  services = {
    rpcbind.enable = true;
    nfs.server = {
      enable = true;
      createMountPoints = true;
      exports = ''
        /nfs         10.0.0.0/24(insecure,rw,sync,no_root_squash,no_subtree_check,crossmnt,fsid=0)
        /nfs/project 10.0.0.0/24(insecure,rw,sync,no_root_squash,no_subtree_check)
        /nfs/scratch 10.0.0.0/24(insecure,rw,sync,no_root_squash,no_subtree_check,nohide)
      '';
      statdPort = 4000;
      lockdPort = 4001;
      mountdPort = 4002;
      extraNfsdConfig = ''
        udp=y
        vers3=on
        vers4=on
      '';
    };
  };

  fileSystems = {
    "/common/project" = {
      depends = [ "/common" "/nfs/project" ];
      device = "/nfs/project";
      fsType = "none";
      options = [ "bind" "nohide" ];
    };
    "/common/scratch" = {
      depends = [ "/common" "/nfs/scratch" ];
      device = "/nfs/scratch";
      fsType = "none";
      options = [ "bind" "nohide" ];
    };
  };
}
