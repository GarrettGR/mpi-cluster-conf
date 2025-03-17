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
          mainNodeIP = clusterConfig.mainNodeIP or "192.168.1.1";
          nodes = clusterConfig.nodes or [
            { name = "mpi-node1"; ip = "192.168.1.11"; isMaster = true; }
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
          
          # Custom package with CUDA-aware MPI
          mkCustomPackages = pkgs: {
            # Custom OpenMPI with CUDA support
            openMPI-cuda = pkgs.openmpi.override {
              enableUCX = true;
              cudaSupport = true;
              cudatoolkit = pkgs.cudaPackages.cudatoolkit;
            };
            
            # Create a sample CUDA-MPI program
            cuda-mpi-hello = pkgs.writeTextFile {
              name = "cuda-mpi-hello.c";
              text = ''
                #include <stdio.h>
                #include <mpi.h>
                #include <cuda_runtime.h>

                __global__ void hello_kernel() {
                    printf("Hello from GPU!\n");
                }

                int main(int argc, char *argv[]) {
                    int rank, size;
                    
                    MPI_Init(&argc, &argv);
                    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
                    MPI_Comm_size(MPI_COMM_WORLD, &size);
                    
                    int deviceCount = 0;
                    cudaGetDeviceCount(&deviceCount);
                    
                    cudaDeviceProp deviceProp;
                    cudaGetDeviceProperties(&deviceProp, 0);
                    
                    printf("MPI Rank %d/%d on %s with GPU: %s (CUDA Capability %d.%d)\n", 
                           rank, size, getenv("HOSTNAME"), 
                           deviceProp.name, deviceProp.major, deviceProp.minor);
                    
                    hello_kernel<<<1,1>>>();
                    cudaDeviceSynchronize();
                    
                    MPI_Finalize();
                    return 0;
                }
              '';
              destination = "/usr/share/doc/cuda-mpi-cluster/examples/cuda-mpi-hello.c";
              executable = false;
            };
            
            # Compile script for the sample program
            compile-cuda-mpi = pkgs.writeShellScriptBin "compile-cuda-mpi" ''
              #!/bin/sh
              set -e
              
              if [ $# -lt 1 ]; then
                echo "Usage: compile-cuda-mpi <source.c> [output_name]"
                echo "Example: compile-cuda-mpi hello.c hello"
                exit 1
              fi
              
              SOURCE="$1"
              OUTPUT="''${2:-$(basename "$1" .c)}"
              
              echo "Compiling $SOURCE to $OUTPUT..."
              ${pkgs.openmpi}/bin/mpicc -o "$OUTPUT" "$SOURCE" -I${pkgs.cudaPackages.cudatoolkit}/include -L${pkgs.cudaPackages.cudatoolkit}/lib -lcudart
              
              echo "Compilation successful! Run with:"
              echo "run-mpi -n <num_processes> ./$OUTPUT"
            '';
            
            # Run script for MPI programs
            run-mpi = pkgs.writeShellScriptBin "run-mpi" ''
              #!/bin/sh
              
              # Use the hostfile for running across nodes
              ${pkgs.openmpi}/bin/mpirun --hostfile /etc/openmpi-hostfile --allow-run-as-root --mca btl_tcp_if_include ${networkConfig.subnet}.0/24 "$@"
            '';
            
            # Generate a simple example for domain decomposition
            domain-decomposition-example = pkgs.writeTextFile {
              name = "domain-decomposition.c";
              text = ''
                #include <stdio.h>
                #include <stdlib.h>
                #include <mpi.h>
                #include <cuda_runtime.h>

                // CUDA kernel to process a subdomain
                __global__ void process_subdomain(float *data, int size) {
                    int idx = blockIdx.x * blockDim.x + threadIdx.x;
                    if (idx < size) {
                        // Simple computation (add 1.0 to each element)
                        data[idx] = data[idx] + 1.0f;
                    }
                }

                int main(int argc, char *argv[]) {
                    int rank, size, i;
                    int domain_size = 1024; // Total size of the domain
                    float *global_data = NULL;
                    float *local_data;
                    
                    MPI_Init(&argc, &argv);
                    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
                    MPI_Comm_size(MPI_COMM_WORLD, &size);
                    
                    // Calculate local subdomain size
                    int local_size = domain_size / size;
                    int remainder = domain_size % size;
                    int my_size = local_size + (rank < remainder ? 1 : 0);
                    int my_offset = rank * local_size + (rank < remainder ? rank : remainder);
                    
                    // Allocate memory for local data
                    local_data = (float*)malloc(my_size * sizeof(float));
                    
                    // Only the root process initializes the global data
                    if (rank == 0) {
                        global_data = (float*)malloc(domain_size * sizeof(float));
                        for (i = 0; i < domain_size; i++) {
                            global_data[i] = (float)i;
                        }
                        printf("Initial data: [%.1f, %.1f, ..., %.1f]\n", 
                               global_data[0], global_data[1], global_data[domain_size-1]);
                    }
                    
                    // Distribute the data
                    MPI_Scatterv(global_data, NULL, NULL, MPI_FLOAT, 
                                local_data, my_size, MPI_FLOAT, 0, MPI_COMM_WORLD);
                    
                    // Process local data on GPU
                    float *d_local_data;
                    cudaMalloc(&d_local_data, my_size * sizeof(float));
                    cudaMemcpy(d_local_data, local_data, my_size * sizeof(float), cudaMemcpyHostToDevice);
                    
                    // Launch kernel (with 256 threads per block)
                    int num_blocks = (my_size + 255) / 256;
                    process_subdomain<<<num_blocks, 256>>>(d_local_data, my_size);
                    
                    // Copy results back
                    cudaMemcpy(local_data, d_local_data, my_size * sizeof(float), cudaMemcpyDeviceToHost);
                    cudaFree(d_local_data);
                    
                    // Print results from each process
                    printf("Rank %d processed elements %d to %d\n", rank, my_offset, my_offset + my_size - 1);
                    
                    // Gather the results back to the root
                    MPI_Gatherv(local_data, my_size, MPI_FLOAT,
                               global_data, NULL, NULL, MPI_FLOAT, 0, MPI_COMM_WORLD);
                    
                    // Print final results
                    if (rank == 0) {
                        printf("Final data: [%.1f, %.1f, ..., %.1f]\n", 
                               global_data[0], global_data[1], global_data[domain_size-1]);
                        free(global_data);
                    }
                    
                    free(local_data);
                    MPI_Finalize();
                    return 0;
                }
              '';
              destination = "/usr/share/doc/cuda-mpi-cluster/examples/domain-decomposition.c";
              executable = false;
            };
            
            # Helper for students to quickly compile and run examples
            run-example = pkgs.writeShellScriptBin "run-example" ''
              #!/bin/sh
              set -e
              
              if [ $# -lt 1 ]; then
                echo "Usage: run-example <example_name> [num_processes]"
                echo "Available examples:"
                echo "  hello - Simple CUDA-MPI hello world"
                echo "  domain - Domain decomposition example"
                exit 1
              fi
              
              EXAMPLE="$1"
              NUM_PROCS="''${2:-2}"
              WORKDIR="$HOME/mpi-examples"
              
              mkdir -p "$WORKDIR"
              cd "$WORKDIR"
              
              case "$EXAMPLE" in
                hello)
                  cp /usr/share/doc/cuda-mpi-cluster/examples/cuda-mpi-hello.c ./
                  compile-cuda-mpi cuda-mpi-hello.c hello
                  run-mpi -n "$NUM_PROCS" ./hello
                  ;;
                domain)
                  cp /usr/share/doc/cuda-mpi-cluster/examples/domain-decomposition.c ./
                  compile-cuda-mpi domain-decomposition.c domain
                  run-mpi -n "$NUM_PROCS" ./domain
                  ;;
                *)
                  echo "Unknown example: $EXAMPLE"
                  exit 1
                  ;;
              esac
            '';
            
            # Create documentation for students
            cuda-mpi-docs = pkgs.writeTextFile {
              name = "cuda-mpi-docs";
              text = ''
                # CUDA-Aware MPI Cluster Documentation

                Welcome to the CUDA-MPI educational cluster! This system is set up
                for learning about CUDA-aware MPI programming and domain decomposition.

                ## Getting Started

                1. **Run a simple example**:
                   ```
                   run-example hello 2
                   ```
                   This will compile and run the hello world example on 2 processes.

                2. **Run the domain decomposition example**:
                   ```
                   run-example domain 4
                   ```
                   This demonstrates basic domain decomposition across 4 processes.

                ## Compiling Your Own Code

                Use the `compile-cuda-mpi` script to compile your CUDA-MPI programs:
                ```
                compile-cuda-mpi your_program.c program_name
                ```

                Then run it with:
                ```
                run-mpi -n <num_processes> ./program_name
                ```

                ## Shared Files

                All nodes share the following directories through NFS:
                - `/home` - Your home directory
                - `/shared` - Shared workspace

                ## Network Configuration

                The cluster uses a ${networkConfig.subnet}.0/24 network.
                Master node: ${masterNode.name} (${masterNode.ip})

                ## Need Help?

                Check the examples in `/usr/share/doc/cuda-mpi-cluster/examples/`
                for reference implementations.
              '';
              destination = "/usr/share/doc/cuda-mpi-cluster/README.md";
              executable = false;
            };
          };
          
          commonConfiguration = { config, pkgs, lib, ... }: 
            let
              customPkgs = mkCustomPackages pkgs;
            in {
              boot.loader.systemd-boot.enable = true;
              boot.loader.efi.canTouchEfiVariables = true;
              
              networking = {
                firewall.enable = false; # Disable firewall for simplicity
                extraHosts = hostEntries;
              };
              
              users.mutableUsers = true;
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
              
              # Add a default user if none specified
              users.users = lib.mkIf (users == []) {
                student = {
                  isNormalUser = true;
                  extraGroups = [ "wheel" "networkmanager" ];
                  # Default password: "student"
                  hashedPassword = "$6$4FxqA0Vy1QhfpBm3$vx5SCRFHgDU.Pc5JM1mXEm8YxpbZRMbUL3tYKPQkYY2qJ7CuGoKRc8Y0ch10S.pKL3/CztMeRGi5oPKNKA7An.";
                };
              };
              
              # Enable sudo without password for simplicity
              security.sudo.wheelNeedsPassword = false;
              
              services.openssh = {
                enable = true;
                settings = {
                  PermitRootLogin = "no";
                  PasswordAuthentication = true;
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
                
                # CUDA and MPI
                cudaPackages.cudatoolkit
                cudaPackages.cuda_cudart
                customPkgs.openMPI-cuda
                
                # Custom scripts and examples
                customPkgs.compile-cuda-mpi
                customPkgs.run-mpi
                customPkgs.run-example
                customPkgs.cuda-mpi-hello
                customPkgs.domain-decomposition-example
                customPkgs.cuda-mpi-docs
                
                # Add user-specified packages
                extraPackages
              ];
              
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
                    home.stateVersion = "23.11";
                    
                    # Default MPI example directory
                    home.file."mpi-examples/.keep".text = "";
                    
                    programs.bash = {
                      enable = true;
                      shellAliases = {
                        ll = "ls -la";
                        ".." = "cd ..";
                      };
                      initExtra = ''
                        # Display welcome message
                        echo "Welcome to the CUDA-MPI Educational Cluster!"
                        echo "Run 'cat /usr/share/doc/cuda-mpi-cluster/README.md' for documentation."
                        echo "Try 'run-example hello 2' to run a simple example."
                        
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
                  home.stateVersion = "23.11";
                  
                  # Default MPI example directory
                  home.file."mpi-examples/.keep".text = "";
                  
                  programs.bash = {
                    enable = true;
                    shellAliases = {
                      ll = "ls -la";
                      ".." = "cd ..";
                    };
                    initExtra = ''
                      # Display welcome message
                      echo "Welcome to the CUDA-MPI Educational Cluster!"
                      echo "Run 'cat /usr/share/doc/cuda-mpi-cluster/README.md' for documentation."
                      echo "Try 'run-example hello 2' to run a simple example."
                      
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
              
              system.stateVersion = "23.11";
            };
          
          masterConfiguration = { config, pkgs, lib, ... }: {
            networking.hostName = masterNode.name;
            networking.interfaces.enp0s3.ipv4.addresses = [
              { address = masterNode.ip; prefixLength = 24; }
            ];
            
            services.nfs.server = {
              enable = true;
              exports = ''
                /home 192.168.1.0/24(rw,sync,no_subtree_check,no_root_squash,insecure)
                /shared 192.168.1.0/24(rw,sync,no_subtree_check,no_root_squash,insecure)
              '';
              extraNfsdConfig = ''
                grace-time=10
              '';
            };
            
            services.rpcbind.enable = true;
            
            systemd.services.nfs-server.serviceConfig = {
              RestartSec = "10s";
              Restart = "on-failure";
            };
            
            networking.networkmanager = {
              enable = true;
              unmanaged = [];
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
            
            networking.networkmanager = {
              enable = true;
              unmanaged = [];
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
