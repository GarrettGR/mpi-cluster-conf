# Instructor's Guide to the CUDA-MPI Educational Cluster

This guide explains how to set up, configure, and manage the CUDA-MPI cluster for educational use.

## Overview

The CUDA-MPI cluster is designed to provide students with a simple environment for learning parallel computing concepts using CUDA and MPI. The system is optimized for:

- Simplicity: Students can focus on learning, not system administration
- Resilience: Built to handle laptop disconnections and reconnections
- Portability: Works with laptops and USB-to-Ethernet adapters

## Setting Up the Cluster

### Prerequisites

- Laptops with NVIDIA GPUs
- USB-to-Ethernet adapters (if laptops don't have Ethernet ports)
- Ethernet switch
- NixOS installation media

### Initial Setup

1. **Install NixOS on each laptop**

   Follow the standard NixOS installation procedure on each laptop.

2. **Clone the cluster repository**

   ```bash
   git clone https://github.com/yourusername/cuda-mpi-cluster
   cd cuda-mpi-cluster
   ```

3. **Configure the cluster**

   Edit `cluster-config.nix` to define your nodes, users, and network configuration:

   ```bash
   nano cluster-config.nix
   ```

   Ensure each laptop has a unique hostname and IP address. The first laptop should be designated as the master node (`isMaster = true`).

4. **Deploy the configuration**

   On each laptop, deploy the NixOS configuration:

   ```bash
   # On the master node
   sudo nixos-rebuild switch --flake .#mpi-main

   # On worker node 1
   sudo nixos-rebuild switch --flake .#mpi-node1

   # On worker node 2
   sudo nixos-rebuild switch --flake .#mpi-node2
   ```

   Replace `mpi-main`, `mpi-node1`, etc. with the hostnames you defined in your configuration.

5. **Verify the setup**

   On the master node, test that all nodes can communicate:

   ```bash
   # Check if NFS shares are exported
   showmount -e localhost

   # Run a simple MPI job across all nodes
   su - student1
   run-example hello 4
   ```

## Network Configuration

### Physical Setup

1. Connect all laptops to the same Ethernet switch
2. If using USB-to-Ethernet adapters, ensure they're properly recognized

### IP Address Assignment

Each laptop needs a static IP address as defined in `cluster-config.nix`. The networking is configured automatically by the NixOS configuration.

### Troubleshooting Network Issues

If nodes can't communicate:

1. Check physical connections
2. Verify IP addresses:
   ```bash
   ip addr show
   ```
3. Verify hostnames and IP mapping:
   ```bash
   cat /etc/hosts
   ```
4. Try pinging between nodes:
   ```bash
   ping mpi-node1
   ```

## Managing Users

### Adding New Users

To add a new user, edit `cluster-config.nix` and add a new entry to the `users` list:

```nix
users = [
  # Existing users...
  {
    name = "newstudent";
    password = "password";
    groups = [ "wheel" "networkmanager" ];
    homeConfig = { pkgs, ... }: {
      home.packages = with pkgs; [ python3 ];
    };
  }
];
```

Then redeploy the configuration on all nodes.

### Generating SSH Keys

For passwordless SSH between nodes:

1. Generate keys for a user:
   ```bash
   sudo su - student1
   ssh-keygen -t ed25519
   ```

2. Add the public key to `cluster-config.nix`:
   ```nix
   users = [
     {
       name = "student1";
       # Other settings...
       sshKeys = [
         "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAI... student1@mpi-main"
       ];
     }
   ];
   ```

3. Redeploy the configuration on all nodes.

## NFS Configuration

The master node automatically exports `/home` and `/shared` directories to all other nodes. 

### Handling Disconnections

The NFS mounts are configured to be resilient to disconnections:

- `soft` option: prevents clients from hanging indefinitely on disconnection
- `timeo` and `retrans`: control timeout behavior
- `x-systemd.automount`: automatically remounts when available
- `autofs`: provides additional resilience for network mounts

If a student's laptop disconnects and reconnects, the NFS shares should automatically reconnect. If not, they can try:

```bash
sudo systemctl restart autofs
```

## Adding Software

### System-wide Packages

To add software for all users, edit `cluster-config.nix` and add to the `extraPackages` list:

```nix
extraPackages = pkgs: with pkgs; [
  python3
  jupyter
  vscode
  # Add more packages here
];
```

### User-specific Packages

To add packages for a specific user, edit their `homeConfig`:

```nix
homeConfig = { pkgs, ... }: {
  home.packages = with pkgs; [
    python3
    python3Packages.numpy
    python3Packages.matplotlib
    # Add more packages here
  ];
};
```

## Classroom Management

### Preparing for a Lab Session

1. **Verify all hardware is connected properly**
   - Laptops powered on
   - Network connections established
   - GPUs functioning (`nvidia-smi` shows all GPUs)

2. **Test the cluster**
   ```bash
   # Run on master node as a student user
   run-example hello 4
   run-example domain 4
   ```

3. **Reset student work (if needed)**
   ```bash
   # Clear all student work directories
   sudo rm -rf /home/student*/mpi-examples/*
   
   # Reset to examples
   sudo cp -r /usr/share/doc/cuda-mpi-cluster/examples/* /shared/examples/
   ```

### Monitoring Student Progress

1. **Check running jobs**
   ```bash
   # See what MPI processes are running
   ps aux | grep mpirun
   ```

2. **Monitor GPU usage**
   ```bash
   nvidia-smi
   # Or for continuous monitoring
   watch -n 1 nvidia-smi
   ```

## Educational Resources

### Example Assignments

1. **Basic MPI-CUDA Integration**
   - Have students modify the hello world example to perform simple vector operations

2. **1D Domain Decomposition**
   - Implement a parallel vector addition across multiple GPUs

3. **2D Domain Decomposition**
   - Implement a parallel matrix multiplication or stencil operation

4. **Halo Exchange Pattern**
   - Implement a simulation (like heat transfer) requiring neighbor communication

### Performance Analysis

Have students experiment with:

1. Different domain decomposition strategies
2. Various message passing patterns
3. Overlapping computation and communication
4. Strong vs. weak scaling experiments

## Troubleshooting

### Common Issues and Solutions

1. **NFS mount issues**
   ```bash
   # Check NFS server status
   systemctl status nfs-server

   # Check mounts on client
   mount | grep nfs
   ```

2. **MPI communication issues**
   ```bash
   # Check OpenMPI hostfile
   cat /etc/openmpi-hostfile

   # Test basic connectivity
   mpirun --hostfile /etc/openmpi-hostfile hostname
   ```

3. **CUDA problems**
   ```bash
   # Verify NVIDIA driver
   nvidia-smi

   # Check CUDA version
   nvcc --version
   ```

### Handling System Updates

NixOS prevents unexpected changes through its declarative configuration. To update:

1. Update the flake.lock file:
   ```bash
   nix flake update
   ```

2. Rebuild and switch to test update on master node:
   ```bash
   sudo nixos-rebuild switch --flake .#mpi-main
   ```

3. If everything works, deploy to all nodes.

## Advanced Customization

### Adding Alternative Storage Backends

The default setup uses NFS, but you could modify the flake to support other options:

1. Edit `minimal-mpi-cluster-flake.nix`
2. Modify the storage configuration sections for both master and worker nodes

### Adding Monitoring Tools

To add monitoring tools:

1. Add Prometheus and Grafana to the extraPackages
2. Configure them in the `masterConfiguration` section

### Custom CUDA Examples

To add additional CUDA examples:

1. Create new example C files
2. Add them to the `mkCustomPackages` function in the flake
3. Update the run-example script to include your new examples

## Conclusion

This cluster setup provides a simplified environment for teaching CUDA-aware MPI programming. By handling the infrastructure aspects, students can focus on learning parallel programming concepts rather than system administration details.

For more information or to report issues, visit the project repository: https://github.com/yourusername/cuda-mpi-cluster
