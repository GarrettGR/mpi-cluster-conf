{
  nodes = [
    {
      name = "node0";
      ip = "10.204.139.32";
      isMaster = true;
      slots = 16;
    }

    {
      name = "node1";
      ip = "10.204.139.28";
      isMaster = false;
      slots = 8;
    }
    
    {
      name = "node2";
      ip = "10.204.139.31";
      isMaster = false;
      slots = 8;
    }

    {
      name = "node3"; 
      ip = "10.204.139.19";
      isMaster = false;
      slots = 20;
    }
  ];

  networkConfig = {
    domain = "mpicluster.local";
    subnet = "10.204.139";
    netmask = "255.255.255.0";
  };

  users = [
    {
      name = "garrettgr";
      description = "Garrett Gonzalez-Rivas";
      shell = pkgs.zsh; 
      # password = "password";
      hashedPassword = "$y$j9T$aJmECtPF9vQFrrcKekuiC.$GdBTLC1ly84/cIJik7AMhK2iy2lYHLJxvVe3ywu9wr8";
      groups = [ "wheel" "networkmanager" ];
      sshKeys = [
        "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIMdLb7Af2+G0PWH5RzMg7Q2Jxro9xusQ3WufUDgaj1E4"
      ];
      homeConfig = { pkgs, ... }: {
        home.packages = with pkgs; [
          neovim
          yazi
          fzf
          tldr
          bat
          eza
        ];
      };
    },
    {
      name = "abigoz";
      description = "Abi Gail Goz";
      password = "password";
      groups = [ "wheel" "networkmanager" ];
    }
  ];

  extraPackages = pkgs: with pkgs; [
    wget
    tmux
    htop
    vim
  ];
}
