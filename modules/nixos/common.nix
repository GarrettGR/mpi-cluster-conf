{config, pkgs, ...}: {
  boot = {
    loader = {
      systemd-boot.enable = true;
      efi.canTouchEfiVariables = true;
    };
  };

  time.timeZone = "America/New_York";
  i18n.defaultLocale = "en_US.UTF-8";

  users = {
    mutableUsers = false;
    users = {
      mpiuser = {
        isNormalUser = true;
        description = "MPI Cluster User";
        extraGroups = [ "wheel" "networkmanager" ];
        createHome = true;
        # hashedPassword = "$y$j9T$aJmECtPF9vQFrrcKekuiC.$GdBTLC1ly84/cIJik7AMhK2iy2lYHLJxvVe3ywu9wr8";
        # openssh.authorizedKeys.keys = [ "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIMdLb7Af2+G0PWH5RzMg7Q2Jxro9xusQ3WufUDgaj1E4" ];

        packages = with pkgs; [
          fastfetch
          tldr
          bat
        ];
      };
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
    mpich-pmix # only for the compiler wrapper
  ];

  services = {
    openssh.enable = true;
    tailscale.enable = true;
  };

  system.stateVersion = "24.11";

  nix = {
    package = pkgs.nix;
    settings.experimental-features = [ "nix-command" "flakes" ];
  };
}
