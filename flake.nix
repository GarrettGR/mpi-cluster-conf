{
  description = "CUDA-aware MPI Cluster for Education and Research";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    home-manager = {
      url = "github:nix-community/home-manager";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = {
    self,
    nixpkgs,
    home-manager,
  }: let
    loadConfig = configPath:
      if builtins.pathExists configPath
      then import configPath
      else abort "Error: Configuration file ${toString configPath} not found!";

    clusterConfigPath = ./cluster-config.nix;
    clusterConfig = loadConfig clusterConfigPath;

    mkMpiCluster = clusterConfig: let
      lib = nixpkgs.lib;

      networkConfig =
        clusterConfig.networkConfig
        or {
          domain = "local";
          subnet = "192.168.1";
          netmask = "255.255.255.0";
        };

      nodes =
        clusterConfig.nodes
        or [
          {
            name = "mpi-node0";
            ip = "192.168.1.10";
            interface = "enp0s3";
            isMaster = true;
            slots = 4;
          }
          {
            name = "mpi-node1";
            ip = "192.168.1.11";
            interface = "enp0s3";
            isMaster = false;
            slots = 4;
          }
        ];

      users = clusterConfig.users or [];

      rawExtraPackages = clusterConfig.extraPackages or [];
      extraPackages = pkgs:
        if builtins.isList rawExtraPackages && (builtins.length rawExtraPackages == 0 || builtins.isString (builtins.head rawExtraPackages))
        then builtins.map (name: pkgs.${name}) rawExtraPackages
        else rawExtraPackages;

      nfsConfig =
        clusterConfig.nfsConfig
        or {
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

      masterNode = builtins.head (builtins.filter (n: n.isMaster) nodes);

      hostEntries = builtins.concatStringsSep "\n" (
        builtins.map (node: "${node.ip} ${node.name}.${networkConfig.domain} ${node.name}") nodes
      );

      commonConfiguration = {
        config,
        pkgs,
        lib,
        ...
      }: {
        boot.loader.systemd-boot.enable = true;
        boot.loader.efi.canTouchEfiVariables = true;

        nixpkgs.config.allowUnfree = true;

        fileSystems."/" = lib.mkDefault {
          #NOTE: uhhhh... is this right?
          device = "/dev/disk/by-label/nixos";
          fsType = "ext4";
        };

        fileSystems."/boot" = lib.mkDefault {
          #NOTE: uhhhh... is this right?
          device = "/dev/disk/by-label/boot";
          fsType = "vfat";
        };

        networking = {
          networkmanager.enable = true;
          firewall.enable = false; # NOTE: disable firewall for simplicity
          extraHosts = hostEntries;
          nameservers = ["1.1.1.1" "8.8.8.8"];
        };

        users = {
          mutableUsers = false;
          users = lib.mkMerge [
            (lib.mkIf (users != []) (
              builtins.listToAttrs (
                builtins.map (user: {
                  name = user.name;
                  value = {
                    isNormalUser = true;
                    extraGroups = user.groups or ["wheel" "networkmanager"];
                    hashedPassword = user.hashedPassword or null;
                    password = user.password or null;
                    description = user.description or null;
                    shell =
                      if user.shell or null != null
                      then pkgs.${user.shell}
                      else pkgs.bash;
                    openssh.authorizedKeys.keys = user.sshKeys or [];
                  };
                })
                users
              )
            ))

            #NOTE: default student user if no users provided
            (lib.mkIf (users == []) {
              student = {
                isNormalUser = true;
                extraGroups = ["wheel" "networkmanager"];
                # Default password: "student"
                initialHashedPassword = "$6$4FxqA0Vy1QhfpBm3$vx5SCRFHgDU.Pc5JM1mXEm8YxpbZRMbUL3tYKPQkYY2qJ7CuGoKRc8Y0ch10S.pKL3/CztMeRGi5oPKNKA7An.";
              };
            })
          ];
        };

        security.sudo.wheelNeedsPassword = false; #FIXME: I really should change this...

        programs.zsh.enable = true;

        services = {
          openssh = {
            enable = true;
            settings = {
              PermitRootLogin = "no";
              PasswordAuthentication = true;
            };
          };

          tailscale.enable = true;
        };

        environment.systemPackages = with pkgs;
          [
            git
            curl
            wget
            vim
            htop
            tmux

            # CUDA tools
            cudaPackages.cudatoolkit
            cudaPackages.cuda_cudart

            # OpenMPI with CUDA support
            openmpi
          ]
          ++ (extraPackages pkgs);

        environment.variables = {
          X_TLS = "rc,sm,cuda_copy,cuda_ipc,gdr_copy";
          UCX_RNDV_SCHEME = "get_zcopy";
          UCX_MEMTYPE_CACHE = "n";
          OMPI_MCA_pml = "ucx";
          OMPI_MCA_btl = "^openib,vader,tcp,uct";
          OMPI_MCA_osc = "ucx";
        };

        hardware.graphics.enable = true;
        hardware.nvidia.package = config.boot.kernelPackages.nvidiaPackages.stable;
        hardware.nvidia.modesetting.enable = true;

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

        home-manager.users = lib.mkMerge [
          (lib.mkIf (users != []) (
            builtins.listToAttrs (
              builtins.map (user: {
                name = user.name;
                value =
                  if user ? homeConfig
                  then user.homeConfig
                  else {
                    home.stateVersion = config.system.stateVersion;

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
                  };
              })
              users
            )
          ))

          #NOTE: default student home configuration if no users provided
          (lib.mkIf (users == []) {
            student = {
              home.stateVersion = config.system.stateVersion;

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
            };
          })
        ];

        time.timeZone = "America/New_York";
        i18n.defaultLocale = "en_US.UTF-8";
        system.stateVersion = "24.11";
      };

      masterConfiguration = {
        config,
        pkgs,
        lib,
        ...
      }: {
        networking.hostName = masterNode.name;
        networking.interfaces.${masterNode.interface or "enp0s3"}.ipv4.addresses = [
          {
            address = masterNode.ip;
            prefixLength = 24;
          }
        ];

        services.nfs.server = {
          enable = true;
          createMountPoints = true;
          statdPort = 4000;
          lockdPort = 4001;
          mountdPort = 4002;

          exports = builtins.concatStringsSep "\n" (
            builtins.map (
              export: "${export.directory} ${networkConfig.subnet}.0/24(${export.options})"
            )
            nfsConfig.exports
          );

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

      workerConfiguration = node: {
        config,
        pkgs,
        lib,
        ...
      }: {
        networking.hostName = node.name;
        networking.interfaces.${node.interface or "enp0s3"}.ipv4.addresses = [
          {
            address = node.ip;
            prefixLength = 24;
          }
        ];
        networking.defaultGateway = masterNode.ip;

        fileSystems = lib.mkMerge [
          (lib.listToAttrs (
            builtins.map (export: {
              name = export.directory;
              value = {
                device = "${masterNode.ip}:${export.directory}";
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
            })
            nfsConfig.exports
          ))
        ];

        systemd.tmpfiles.rules =
          builtins.map (
            export: "d ${export.directory} 0755 root root -"
          )
          nfsConfig.exports;
      };

      nodeConfigurations = builtins.listToAttrs (
        builtins.map (node: {
          name = node.name;
          value = nixpkgs.lib.nixosSystem {
            system = "x86_64-linux";
            modules = [
              commonConfiguration
              (
                if node.isMaster
                then masterConfiguration
                else workerConfiguration node
              )
              home-manager.nixosModules.home-manager
            ];
          };
        })
        nodes
      );
    in
      nodeConfigurations;
  in {
    nixosConfigurations = mkMpiCluster clusterConfig;

    lib = {
      inherit mkMpiCluster;
    };

    checks =
      builtins.mapAttrs (name: value: value.config.system.build.toplevel)
      (mkMpiCluster clusterConfig);
  };
}
