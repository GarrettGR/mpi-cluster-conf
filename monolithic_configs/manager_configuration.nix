{ config, pkgs, ... }:

{
  imports = [ ./hardware-configuration.nix ];

  boot.loader.systemd-boot.enable = true;
  boot.loader.efi.canTouchEfiVariables = true;

  hardware.graphics.enable = true;

  hardware.nvidia = {
    package = config.boot.kernelPackages.nvidiaPackages.production; # installs 550
    modesetting.enable = true;
    nvidiaSettings = true;
  };

  # systemd.network.networks.link."enp0s13f0u2" = {
  #   matchConfig.PermanentMACAddress = "00:e0:4c:96:7e:41";
  #   linkConfig.Name = "eth0";
  # };

  networking.hostName = "node0";
  networking.defaultGateway = "192.168.1.1";
  networking.nameservers = [ "8.8.8.8" ];
  networking.interfaces.enp0s13f0u2.ipv4.addresses = [
    { address = "10.0.0.1"; prefixLength = 24; }
    { address = "192.168.1.201"; prefixLength = 24; }
  ];
  networking.hosts = {
    "10.0.0.1" = ["node0"];
    "10.0.0.2" = ["node1"];
    "10.0.0.3" = ["node2"];
  };

  networking.firewall.enable = false;
  # networking.firewall.allowedTCPPorts = [ 22 111 2049 4000 4001 4002 20048 ];
  # networking.firewall.allowedUDPPorts = [ 111 2049 4000 4001 4002 20048 ];

  networking.networkmanager.enable = true;

  time.timeZone = "America/New_York";

  i18n.defaultLocale = "en_US.UTF-8";

  i18n.extraLocaleSettings = {
    LC_ADDRESS = "en_US.UTF-8";
    LC_IDENTIFICATION = "en_US.UTF-8";
    LC_MEASUREMENT = "en_US.UTF-8";
    LC_MONETARY = "en_US.UTF-8";
    LC_NAME = "en_US.UTF-8";
    LC_NUMERIC = "en_US.UTF-8";
    LC_PAPER = "en_US.UTF-8";
    LC_TELEPHONE = "en_US.UTF-8";
    LC_TIME = "en_US.UTF-8";
  };

  services.xserver.xkb = {
    layout = "us";
    variant = "";
  };

  services.xserver.videoDrivers = [ "nvidia" ];

  systemd.sysusers.enable = false;
  users.mutableUsers = false;
  users.users = {
    garrettgr = {
      isNormalUser = true;
      shell = pkgs.zsh;
      description = "Garrett Gonzalez-Rivas";
      extraGroups = [ "networkmanager" "wheel" ];
      hashedPassword = "$y$j9T$aJmECtPF9vQFrrcKekuiC.$GdBTLC1ly84/cIJik7AMhK2iy2lYHLJxvVe3ywu9wr8";
      openssh.authorizedKeys.keys = [ "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIMdLb7Af2+G0PWH5RzMg7Q2Jxro9xusQ3WufUDgaj1E4" ];
      packages = with pkgs; [
        neovim
        zoxide
        atuin
        fzf
        eza
        bat
        xclip
        yazi
        tldr
        iperf3
        fastfetch
      ];
    };
    abigoz = {
      isNormalUser = true;
      description = "Abi Gail Goz";
      extraGroups = [ "networkmanager" "wheel" ];
      initialHashedPassword = "$y$j9T$7l6jFhuxdtu7U3kROs4Ov.$ryG5haREh6h0wZCLiBulqC1Owhvw4kbnQAMOOnNC0/3";
      packages = with pkgs; [];
    };
    mpiuser = {
      isSystemUser = true;
      # linger = true;
      createHome = false;
      group = "root"; #! uhhhh maybe ??
      extraGroups = [ "wheel" ];
      # packages = with pkgs; [ mpi ];
      openssh.authorizedKeys.keys = [ "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIPz9zqlCv4w8L0nb3kdvW8+c3htkrA8b+KYX+JPWR+8P" ];
    };
  };

  nixpkgs.config.allowUnfree = true;

  environment.systemPackages = with pkgs; [
    vim
    wget
    tmux
    htop
    gcc12
    mpi
    mpich-pmix
    # dtcmp 
    # mpifileutils
    # lwgrp
    # parmetis
    # cudaPackages.nvidia_driver
    cudaPackages.nccl
    # cudaPackages.libcusparse
    # cudaPackages.libcusolver
    # cudaPackages.libcurand
    # cudaPackages.libcublas
    # cudaPackages.cutensor
    cudaPackages.cudatoolkit # installs 12.4
    cudaPackages.cuda_opencl
    # cudaPackages.cuda_profiler_api
    # cudaPackages.cuda_nvprof
    # cudaPackages.cuda_nvprune
    cudaPackages.cuda_nvcc
    cudaPackages.cuda_gdb
    cudaPackages.cuda_cudart
    cudaPackages.backendStdenv
    nvtopPackages.nvidia
  ];

  programs.zsh.enable = true;

  services.openssh.enable = true;
  services.tailscale.enable = true;
  services.rpcbind.enable = true; # do I even need this on the server ??

  # fileSystems = {
  #   "/nfs/project" = {
  #     device = "/mnt/fast-drive";
  #     options = [ "bind" ];
  #   };
  #   "/nfs/scratch" = {
  #     device = "/mnt/bulk-drive";
  #     options = [ "bind" ];
  #   };
  # };

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

  services.nfs.server.enable = true;
  services.nfs.server.createMountPoints = true;
  services.nfs.server.exports = ''
    /nfs 		10.0.0.0/24(insecure,rw,sync,no_root_squash,no_subtree_check,crossmnt,fsid=0)
    /nfs/project 	10.0.0.0/24(insecure,rw,sync,no_root_squash,no_subtree_check)
    /nfs/scratch	10.0.0.0/24(insecure,rw,sync,no_root_squash,no_subtree_check,nohide)
  '';
  services.nfs.server.statdPort  = 4000;
  services.nfs.server.lockdPort  = 4001;
  services.nfs.server.mountdPort = 4002;
  services.nfs.server.extraNfsdConfig = ''
    udp=y
    vers3=on
    vers4=on
  '';

  nix.package = pkgs.lix;
  nix.settings.experimental-features = [ "nix-command" "flakes" ];

  system.stateVersion = "24.11";
}
