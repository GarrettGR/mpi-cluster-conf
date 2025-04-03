{
  networkConfig = {
    domain = "mpicluster.local";
    subnet = "10.204.139";
    netmask = "255.255.255.0";
  };

  nodes = [
    {
      name = "node0";
      ip = "10.204.139.32";
      interface = "enp0s3";
      isMaster = true;
      slots = 16;
    }
    {
      name = "node1";
      ip = "10.204.139.28";
      interface = "enp0s3";
      isMaster = false;
      slots = 8;
    }
    {
      name = "node2";
      ip = "10.204.139.31";
      interface = "enp0s3";
      isMaster = false;
      slots = 8;
    }
    {
      name = "node3";
      ip = "10.204.139.19";
      interface = "enp0s3";
      isMaster = false;
      slots = 20;
    }
  ];

  users = [
    {
      name = "garrettgr";
      description = "Garrett Gonzalez-Rivas";
      hashedPassword = "$y$j9T$aJmECtPF9vQFrrcKekuiC.$GdBTLC1ly84/cIJik7AMhK2iy2lYHLJxvVe3ywu9wr8";
      groups = ["wheel" "networkmanager"];
      sshKeys = [
        "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIMdLb7Af2+G0PWH5RzMg7Q2Jxro9xusQ3WufUDgaj1E4"
      ];
      shell = "zsh";
      homeConfig = {pkgs, ...}: {
        home.stateVersion = "24.11";
        home.packages = with pkgs; [
          neovim
          yazi
          fzf
          tldr
          bat
          eza
        ];
      };
    }

    {
      name = "abigoz";
      description = "Abi Gail Goz";
      password = "password";
      groups = ["wheel" "networkmanager"];
    }
  ];

  extraPackages = [
    "wget"
    "tmux"
    "htop"
    "vim"
  ];

  nfsConfig = {
    exports = [
      {
        directory = "/home";
        options = "rw,sync,no_subtree_check,no_root_squash,insecure";
      }

      {
        directory = "/shared";
        options = "rw,sync,no_subtree_check,no_root_squash,insecure";
      }
    ];
  };
}
