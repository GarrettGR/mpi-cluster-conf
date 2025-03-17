{ sshKeys }:

{ config, pkgs, lib, ... }: {
  boot.loader.systemd-boot.enable = true;
  boot.loader.efi.canTouchEfiVariables = true;

  users = {
    mutableUsers = false;
    users.root.openssh.authorizedKeys.keys = sshKeys;
    users.admin = {
      isNormalUser = true;
      extraGroups = [ "wheel" ];
      openssh.authorizedKeys.keys = sshKeys;
    };
    users.slurm = {
      isNormalUser = true;
      group = "slurm";
      uid = 1000;
      home = "/var/lib/slurm";
      createHome = true;
    };
  };
  
  security.sudo.wheelNeedsPassword = false;

  services.openssh = {
    enable = true;
    settings = {
      PermitRootLogin = "prohibit-password";
      PasswordAuthentication = false;
    };
  };

  services.timesyncd.enable = true;

  environment.systemPackages = with pkgs; [
    vim
    git
    htop
    tmux
    wget
    curl
    rsync
    lm_sensors
    sysstat
    nvtop
    hwloc
    ethtool
    iperf
    screen
    jq
    parted
    gptfdisk
    nvme-cli
  ];

  system.activationScripts.generateSshKey = ''
    if [ ! -f /root/.ssh/id_ed25519 ]; then
      ${pkgs.openssh}/bin/ssh-keygen -t ed25519 -N "" -f /root/.ssh/id_ed25519
    fi
  '';

  services.journald.extraConfig = ''
    SystemMaxUse=1G
    MaxRetentionSec=1week
  '';
  
  nix = {
    package = pkgs.nixVersions.stable;
    extraOptions = ''
      experimental-features = nix-command flakes
    '';
    settings = {
      auto-optimise-store = true;
      trusted-users = [ "root" "admin" ];
    };
    gc = {
      automatic = true;
      dates = "weekly";
      options = "--delete-older-than 30d";
    };
  };
}
