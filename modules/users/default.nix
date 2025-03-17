# modules/users/default.nix - User management with Home Manager integration

{ home-manager }:

{
  # Main function to create user configuration module
  mkModule = { users, node, nodes, sshKeys }:
    { config, pkgs, lib, ... }:
    
    let
      isController = node.isController or false;
      
      # Function to filter users based on node type and user's allowed nodes
      nodeUsers = builtins.filter (user: 
        (user.nodes == null) || 
        (user.nodes == "all") || 
        (user.nodes == "controller" && isController) || 
        (user.nodes == "workers" && !isController) || 
        (builtins.elem node.hostname (if builtins.isList user.nodes then user.nodes else []))
      ) users;
      
      # Generate Home Manager configurations for all users on this node
      hmConfigurations = map (user: {
        name = user.name;
        value = {
          home-manager.users.${user.name} = user.homeConfig or {};
          users.users.${user.name} = {
            isNormalUser = true;
            extraGroups = user.groups or [];
            openssh.authorizedKeys.keys = user.sshKeys or sshKeys;
          } // (removeAttrs user ["name" "homeConfig" "groups" "sshKeys" "nodes"]);
        };
      }) nodeUsers;
      
    in lib.mkMerge ([
      # Add Home Manager support
      ({ imports = [ home-manager.nixosModules.home-manager ]; })
      
      # Add base Home Manager configuration
      {
        home-manager = {
          useGlobalPkgs = true;
          useUserPackages = true;
          
          # Default configuration for all users
          sharedModules = [
            ({ config, pkgs, ... }: {
              # Common home-manager configurations for all users
              programs.bash = {
                enable = true;
                shellAliases = {
                  ls = "ls --color=auto";
                  ll = "ls -la";
                  ".." = "cd ..";
                };
                initExtra = ''
                  # Default HPC environment variables
                  export PATH=$HOME/bin:$PATH
                  
                  # SLURM shortcuts
                  alias si='sinfo'
                  alias sq='squeue'
                  alias sa='sacct'
                  
                  # Show a custom prompt with hostname in color
                  PS1='\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ '
                '';
              };
              
              # Default editor configuration
              programs.vim = {
                enable = true;
                settings = {
                  number = true;
                  tabstop = 4;
                  shiftwidth = 4;
                };
                extraConfig = ''
                  syntax on
                  set expandtab
                  set autoindent
                  set hlsearch
                  set incsearch
                  set mouse=a
                '';
              };
              
              # Common Git configuration
              programs.git = {
                enable = true;
                extraConfig = {
                  pull.rebase = false;
                  init.defaultBranch = "main";
                };
              };
              
              # Common directories
              home.file = {
                ".local/bin/.keep".text = "";
                "projects/.keep".text = "";
                "data/.keep".text = "";
              };
            })
          ];
        };
      }
    ] 
    # Add individual user configurations
    ++ (map (hm: hm.value) hmConfigurations));
}
