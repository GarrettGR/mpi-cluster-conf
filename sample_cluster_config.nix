# cluster-config.nix - Configuration for CUDA-MPI Educational Cluster
# This file defines the cluster setup for student use

{
  # Define all cluster nodes
  nodes = [
    # First laptop is the master node (handles NFS sharing)
    { 
      name = "mpi-main";  # Hostname
      ip = "192.168.10.1"; # IP address
      isMaster = true;     # This is the master node
      slots = 4;           # Number of MPI slots (processes) to allocate
    }
    # Second laptop
    { 
      name = "mpi-node1"; 
      ip = "192.168.10.2";
      isMaster = false;
      slots = 4;
    }
    # Third laptop (optional)
    # { 
    #   name = "mpi-node2"; 
    #   ip = "192.168.10.3";
    #   isMaster = false;
    #   slots = 4;
    # }
    # Fourth laptop (optional)
    # { 
    #   name = "mpi-node3"; 
    #   ip = "192.168.10.4";
    #   isMaster = false;
    #   slots = 4;
    # }
  ];

  # Network configuration
  networkConfig = {
    domain = "mpicluster.local";  # Local domain name
    subnet = "192.168.10";        # First three octets of IP addresses
    netmask = "255.255.255.0";    # Subnet mask
  };

  # User accounts
  users = [
    {
      name = "student1";
      password = "password1";     # Plain text password (will be hashed) - OR use hashedPassword
      # hashedPassword = "$6$..."; # Pre-hashed password (generate with mkpasswd)
      groups = [ "wheel" "networkmanager" ]; # wheel for sudo access
      sshKeys = [
        # "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIMMMMMMMMMMMMMMMMMM student1@laptop"
      ];
      # Custom Home Manager configuration
      homeConfig = { pkgs, ... }: {
        home.packages = with pkgs; [
          python3
          python3Packages.numpy
          python3Packages.matplotlib
        ];
        
        # Create example MPI program
        home.file."mpi-examples/hello.c".text = ''
          #include <stdio.h>
          #include <mpi.h>

          int main(int argc, char** argv) {
              MPI_Init(&argc, &argv);

              int world_size;
              MPI_Comm_size(MPI_COMM_WORLD, &world_size);

              int world_rank;
              MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

              char processor_name[MPI_MAX_PROCESSOR_NAME];
              int name_len;
              MPI_Get_processor_name(processor_name, &name_len);

              printf("Hello world from processor %s, rank %d out of %d processors\n",
                     processor_name, world_rank, world_size);

              MPI_Finalize();
              return 0;
          }
        '';
      };
    },
    {
      name = "student2";
      password = "password2";
      groups = [ "wheel" "networkmanager" ];
    }
  ];

  # Additional packages to install on all nodes
  extraPackages = pkgs: with pkgs; [
    python3
    jupyter
    vscode
    firefox
    thunderbird
  ];

  # SSH keys for all users (optional)
  sshKeys = [
    # "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIMMMMMMMMMMMMMMMMMM instructor@laptop"
  ];
}
