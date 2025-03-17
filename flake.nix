{
  description = "Modular NixOS HPC Cluster with SLURM, OpenMPI, CUDA, and shared storage";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    home-manager = {
      url = "github:nix-community/home-manager";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, flake-utils, home-manager, ... }:
    let 
      lib = import ./lib { inherit nixpkgs; };
      modules = {
        base = import ./modules/base;
        slurm = import ./modules/slurm;
        filesystems = import ./modules/filesystems;
        hpc = import ./modules/hpc;
        networking = import ./modules/networking;
        monitoring = import ./modules/monitoring;
        users = import ./modules/users { inherit home-manager; };
      };
      
      mkClusterFlake = { 
        controllerHostname ? "controller",
        controllerIP ? "10.0.0.1",
        workerNodes ? [
          # Default configuration for a minimal 2-node cluster
          { hostname = "worker1"; ip = "10.0.0.11"; hasGPU = true; cpus = 64; memoryMB = 196608; gpus = 4; }
        ],
        networkConfig ? {
          domain = "cluster.local";
          subnet = "10.0.0";
          netmask = "255.255.255.0";
        },
        filesystemConfig ? {
          type = "nfs"; # NOTE: can be "nfs", "beegfs", etc.
          sharedMounts = [
            { mountPoint = "/home"; exportPath = "/home"; }
            { mountPoint = "/opt/shared"; exportPath = "/opt/shared"; }
          ];
        },

        # Optional SSH keys
        sshKeys ? [
          "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM user@example"
        ],
        system ? "x86_64-linux",
        # Optional user configurations with home-manager
        users ? [],
        # Additional node-specific modules
        extraNodeModules ? {},
        # Additional common modules for all nodes
        extraCommonModules ? []
      }: 
      let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            cudaSupport = true;
          };
        };

        nodes = lib.generateNodes {
          inherit controllerHostname controllerIP workerNodes;
        };

        commonConfig = { config, pkgs, lib, ... }: {
          imports = [
            (modules.base { inherit sshKeys; })
          ];
        };

        mkSystem = name: node:
          nixpkgs.lib.nixosSystem {
            inherit system;
            specialArgs = { inherit pkgs node nodes networkConfig filesystemConfig; };
            modules = [
              commonConfig
              (modules.networking { inherit node nodes networkConfig; })
              (modules.slurm { inherit node nodes; })
              (modules.filesystems { inherit node nodes filesystemConfig; })
              (modules.hpc { inherit node; })
              (modules.monitoring { inherit node nodes; })
              # Add user management module
              (modules.users.mkModule { inherit users node nodes sshKeys; })
              # Add extra modules specific to this node type, if any
              (extraNodeModules.${name} or {})
              # Add any custom system configuration for this node
              node.systemConfig
            ] ++ extraCommonModules;
          };

      in {
        nixosConfigurations = nixpkgs.lib.mapAttrs mkSystem nodes;
        
        packages.${system}.default = pkgs.writeShellScriptBin "cluster-tools" ''
          #!/bin/bash
          
          SCRIPT_NAME=$(basename "$0")
          
          function print_usage {
            echo "NixOS HPC Cluster Management Tools"
            echo ""
            echo "Usage: $SCRIPT_NAME [command]"
            echo ""
            echo "Commands:"
            echo "  generate-munge-key    Generate a new munge key for authentication"
            echo "  list-nodes            List all nodes in the cluster"
            echo "  test-cuda-mpi         Create a test program for CUDA-aware MPI"
            echo "  help                  Show this help message"
            echo ""
          }
          
          function generate_munge_key {
            ${pkgs.munge}/bin/mungekey -c -f -k munge.key
            echo "Generated munge.key - distribute this file to all nodes at /etc/munge/munge.key"
          }
          
          function list_nodes {
            echo "Cluster nodes:"
            echo ""
            echo "Controller: ${controllerHostname} (${controllerIP})"
            echo ""
            echo "Worker nodes:"
            ${pkgs.lib.concatMapStrings (node: 
              ''echo "  ${node.hostname} (${node.ip}) - CPUs: ${toString node.cpus}, Memory: ${toString node.memoryMB}MB${if node.hasGPU then ", GPUs: ${toString node.gpus}" else ""}"''
            ) (pkgs.lib.filter (node: !node.isController) (pkgs.lib.attrValues nodes))}
          }
          
          function test_cuda_mpi {
            cat > cuda_mpi_test.c << 'EOF'
            #include <stdio.h>
            #include <mpi.h>
            #include <cuda_runtime.h>

            int main(int argc, char *argv[]) {
                int rank, size;
                int deviceCount;
                cudaDeviceProp prop;
                
                MPI_Init(&argc, &argv);
                MPI_Comm_rank(MPI_COMM_WORLD, &rank);
                MPI_Comm_size(MPI_COMM_WORLD, &size);
                
                cudaGetDeviceCount(&deviceCount);
                
                if (cudaGetDeviceProperties(&prop, rank % deviceCount) != cudaSuccess) {
                    printf("Error getting device properties\n");
                    return 1;
                }
                
                printf("Process %d/%d running on %s, using GPU %d: %s\n", 
                      rank, size, 
                      getenv("HOSTNAME") ? getenv("HOSTNAME") : "unknown", 
                      rank % deviceCount, 
                      prop.name);
                      
                MPI_Finalize();
                return 0;
            }
            EOF
            
            echo "Compiled CUDA-MPI test program. Compile with:"
            echo "mpicc -o cuda_mpi_test cuda_mpi_test.c -lcudart"
            echo ""
            echo "Run with:"
            echo "mpirun -np 4 --hostfile /etc/openmpi-hostfile ./cuda_mpi_test"
          }
          
          if [ $# -eq 0 ]; then
            print_usage
            exit 1
          fi
          
          COMMAND="$1"
          shift
          
          case "$COMMAND" in
            generate-munge-key)
              generate_munge_key "$@"
              ;;
            list-nodes)
              list_nodes "$@"
              ;;
            test-cuda-mpi)
              test_cuda_mpi "$@"
              ;;
            help)
              print_usage
              ;;
            *)
              echo "Unknown command: $COMMAND"
              print_usage
              exit 1
              ;;
          esac
        '';
      };
      
    in {
      inherit mkClusterFlake;
      
      lib = lib;
      
      # Pre-defined cluster example configurations
      clusters = {
        twoNode = mkClusterFlake {};
        fourNode = mkClusterFlake {
          workerNodes = [
            { hostname = "worker1"; ip = "10.0.0.11"; hasGPU = true; cpus = 64; memoryMB = 196608; gpus = 4; }
            { hostname = "worker2"; ip = "10.0.0.12"; hasGPU = true; cpus = 64; memoryMB = 196608; gpus = 4; }
            { hostname = "worker3"; ip = "10.0.0.13"; hasGPU = true; cpus = 64; memoryMB = 196608; gpus = 4; }
          ];
        };
        beegfs = mkClusterFlake {
          filesystemConfig = {
            type = "beegfs";
            sharedMounts = [
              { mountPoint = "/home"; beegfsDataTarget = "home"; }
              { mountPoint = "/scratch"; beegfsDataTarget = "scratch"; }
            ];
            serverNodes = [ "worker1" ];
          };
        };
      };
      
      modules = modules;
    };
}
