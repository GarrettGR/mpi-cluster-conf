{
  description = "Simple CUDA-aware MPI Cluster for Education";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    home-manager = {
      url = "github:nix-community/home-manager";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, home-manager }:
    let
      clusterConfigPath = ./cluster-config.nix;
      clusterConfig = import clusterConfigPath; # TODO: print an error if the file isn't found??
      
      mkMpiCluster = clusterConfig: 
        let
          hostName = clusterConfig.hostName or "mpi-node";
          mainNode = clusterConfig.mainNode or "mpi-main";
          mainNodeIP = clusterConfig.mainNodeIP or "192.168.1.10";
          nodes = clusterConfig.nodes or [
            { name = "mpi-node0"; ip = "192.168.1.10"; isMaster = true; }
            { name = "mpi-node1"; ip = "192.168.1.11"; isMaster = false; }
            { name = "mpi-node2"; ip = "192.168.1.12"; isMaster = false; }
          ];
          users = clusterConfig.users or [];
          extraPackages = clusterConfig.extraPackages or [];
          networkConfig = clusterConfig.networkConfig or {
            domain = "local";
            subnet = "192.168.1";
            netmask = "255.255.255.0";
          };
          sshKeys = clusterConfig.sshKeys or [];
          
          masterNode = builtins.head (builtins.filter (n: n.isMaster) nodes);
          
          hostEntries = builtins.concatStringsSep "\n" (
            builtins.map (node: "${node.ip} ${node.name}.${networkConfig.domain} ${node.name}") nodes
          );

          commonConfiguration = { config, pkgs, lib, ... }: 
            {

              # NOTE: is this the right way to do this ?? Will this be the node-specific hardware-configuration ??
              # imports = [ /etc/nixos/hardware-configuration.nix ];

              boot.loader.systemd-boot.enable = true;
              boot.loader.efi.canTouchEfiVariables = true;
              
              networking = {
                networkmanager.enable = true;
                firewall.enable = false; # NOTE: Disable firewall for simplicity
                extraHosts = hostEntries;

              systemd.sysusers.enable = false;
              users.mutableUsers = false;
              users.users = builtins.listToAttrs (
                builtins.map (user: {
                  name = user.name;
                  value = {
                    isNormalUser = true;
                    extraGroups = user.groups or [ "wheel" "networkmanager" ];
                    hashedPassword = user.hashedPassword or null;
                    password = user.password or null;
                    openssh.authorizedKeys.keys = user.sshKeys or sshKeys;
                  };
                }) users
              );
              
              users.users = lib.mkIf (users == []) {
                student = {
                  isNormalUser = true;
                  extraGroups = [ "wheel" "networkmanager" ];
                  # Default password: "student"
                  initialHashedPassword = "$6$4FxqA0Vy1QhfpBm3$vx5SCRFHgDU.Pc5JM1mXEm8YxpbZRMbUL3tYKPQkYY2qJ7CuGoKRc8Y0ch10S.pKL3/CztMeRGi5oPKNKA7An.";
                };
              };
              
              # NOTE: Enable sudo without password for simplicity
              security.sudo.wheelNeedsPassword = false;
              
              services = {
                tailscale.enable = true;
                openssh = {
                  enable = true;
                  settings = {
                    PermitRootLogin = "no";
                    PasswordAuthentication = true;
                  };
                };
              };
              
              environment.systemPackages = with pkgs; [
                # Basic utilities
                vim
                wget
                git
                htop
                tmux
                screen
                
                # CUDA
                cudaPackages.cudatoolkit
                cudaPackages.cuda_cudart
                
                # Add user-specified packages
                extraPackages
              ];

              environment.variables = {
                X_TLS = "rc,sm,cuda_copy,cuda_ipc,gdr_copy";
                UCX_RNDV_SCHEME = "get_zcopy";
                UCX_MEMTYPE_CACHE = "n";
                OMPI_MCA_pml = "ucx";
                OMPI_MCA_btl = "^openib,vader,tcp,uct";
                OMPI_MCA_osc = "ucx";
              };
              
              hardware.opengl.enable = true;
              hardware.nvidia.package = config.boot.kernelPackages.nvidiaPackages.stable;
              hardware.nvidia.modesetting.enable = true;
              
              # Create hostfile for OpenMPI
              environment.etc."openmpi-hostfile".text = builtins.concatStringsSep "\n" (
                builtins.map (node: "${node.name} slots=${toString (node.slots or 1)}") nodes
              );
              
              services.autofs = {
                enable = true;
                autoMaster = ''
                  /net -hosts --timeout=60
                '';
              };
              
              system.activationScripts.createSharedDir = ''
                mkdir -p /shared
                chmod 777 /shared
              '';

              home-manager.useGlobalPkgs = true;
              home-manager.useUserPackages = true;
              
              home-manager.users = builtins.listToAttrs (
                builtins.map (user: {
                  name = user.name;
                  value = user.homeConfig or {
                    home.stateVersion = "24.11";
                    
                    programs.bash = {
                      enable = true;
                      shellAliases = {
                        ll = "ls -la";
                        ".." = "cd ..";
                      };
                      initExtra = ''
                        # Add useful environment variables
                        export PATH=$HOME/bin:$PATH
                        export CUDA_HOME=${pkgs.cudaPackages.cudatoolkit}
                      '';
                    };
                    
                    programs.vim = {
                      enable = true;
                      settings = {
                        number = true;
                      };
                      extraConfig = ''
                        syntax on
                        set expandtab
                        set tabstop=4
                        set shiftwidth=4
                      '';
                    };
                  };
                }) users
              );
              
              home-manager.users = lib.mkIf (users == []) {
                student = {
                  home.stateVersion = "24.11";
                  
                  programs.bash = {
                    enable = true;
                    shellAliases = {
                      ll = "ls -la";
                      ".." = "cd ..";
                    };
                    initExtra = ''
                      # Add useful environment variables
                      export PATH=$HOME/bin:$PATH
                      export CUDA_HOME=${pkgs.cudaPackages.cudatoolkit}
                    '';
                  };
                  
                  programs.vim = {
                    enable = true;
                    settings = {
                      number = true;
                    };
                    extraConfig = ''
                      syntax on
                      set expandtab
                      set tabstop=4
                      set shiftwidth=4
                    '';
                  };
                };
              };
          
              time.timeZone = "America/New_York";
              i18n.defaultLocale = "en_US.UTF-8";
    
              system.stateVersion = "24.11";
            };
          
          masterConfiguration = { config, pkgs, lib, ... }: {
            networking.hostName = masterNode.name;
            networking.interfaces.enp0s3.ipv4.addresses = [
              { address = masterNode.ip; prefixLength = 24; }
            ];
            
            services.nfs.server = {
              enable = true;
              createMountPoints = true;
              statdPort  = 4000;
              lockdPort  = 4001;
              mountdPort = 4002;
              exports = ''
                /home 192.168.1.0/24(rw,sync,no_subtree_check,no_root_squash,insecure)
                /shared 192.168.1.0/24(rw,sync,no_subtree_check,no_root_squash,insecure)
              '';
              extraNfsdConfig = ''
                udp=y
                vers3=on
                vers4=on
                grace-time=10
              '';
            };
            
            services.rpcbind.enable = true;
            
            systemd.services.nfs-server.serviceConfig = {
              RestartSec = "10s";
              Restart = "on-failure";
            };
            
            boot.kernel.sysctl = {
              "net.ipv4.ip_forward" = 1;
            };
          };
          
          workerConfiguration = node: { config, pkgs, lib, ... }: {
            networking.hostName = node.name;
            networking.interfaces.enp0s3.ipv4.addresses = [
              { address = node.ip; prefixLength = 24; }
            ];
            networking.defaultGateway = masterNode.ip;
            
            fileSystems."/home" = {
              device = "${masterNode.ip}:/home";
              fsType = "nfs";
              options = [ 
                "noatime" 
                "soft" 
                "timeo=900" 
                "retrans=5" 
                "x-systemd.automount" 
                "x-systemd.idle-timeout=1800" 
                "x-systemd.device-timeout=5s" 
                "x-systemd.mount-timeout=5s" 
              ];
            };
            
            fileSystems."/shared" = {
              device = "${masterNode.ip}:/shared";
              fsType = "nfs";
              options = [ 
                "noatime" 
                "soft" 
                "timeo=900" 
                "retrans=5" 
                "x-systemd.automount" 
                "x-systemd.idle-timeout=1800" 
                "x-systemd.device-timeout=5s" 
                "x-systemd.mount-timeout=5s" 
              ];
            };
            
            systemd.tmpfiles.rules = [
              "d /home 0755 root root -"
              "d /shared 0777 root root -"
            ];
          };
          
          nodeConfigurations = builtins.listToAttrs (
            builtins.map (node: {
              name = node.name;
              value = nixpkgs.lib.nixosSystem {
                system = "x86_64-linux";
                modules = [
                  commonConfiguration
                  (if node.isMaster then masterConfiguration else workerConfiguration node)
                  home-manager.nixosModules.home-manager
                ];
              };
            }) nodes
          );
          
        in nodeConfigurations;

    in {
      nixosConfigurations = mkMpiCluster clusterConfig;
      
      lib = {
        inherit mkMpiCluster;
      };
    };
}
