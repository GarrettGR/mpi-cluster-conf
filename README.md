# NixOS MPI Cluster Setup

This repository contains configuration and setup scripts for creating a small CUDA-aware MPI cluster using NixOS. 

## What is this?

This is a complete setup for creating a small computer cluster that can:
- Run parallel programs using MPI (Message Passing Interface)
- Use NVIDIA GPUs for computation (CUDA)
- Share files between computers using NFS (Network File System)
- Automatically configure itself using NixOS

The cluster consists of:
- One manager node (the main computer that coordinates everything)
- One or more worker nodes (computers that help with calculations)

## Prerequisites

You'll need:
- At least two computers with:
  - Intel processors
  - NVIDIA GPUs
  - Netowrk interfaces (preferably ethernet ports or adapters)
    - An ethernet switch or router to connect the computers
    - Ethernet cables
- USB drives (for installation)
- Basic familiarity with using a terminal

## Step-by-Step Setup Guide

### 1. Physical Setup

1. Connect all computers to the same network switch/router using ethernet cables
2. Make note of which computer will be your manager node (pick one!)
3. Power on all computers

### 2. Network Planning

Before installation, you need to plan your network. Here's the default setup (modify these in `modules/nixos/networking.nix` if needed):

- Manager node (node0):
  - Cluster network: 10.0.0.1
  - External network: 192.168.1.201

- Worker node (node1):
  - Cluster network: 10.0.0.2
  - External network: 192.168.1.202

You'll need to modify these addresses if:
- Your local network uses a different scheme than 192.168.1.x
- You're setting up more than one worker node
- Your network administrator assigned you different addresses

### 3. Installing NixOS

On each computer:

1. Download the NixOS 24.11 ISO from [nixos.org](https://nixos.org/download)
2. Create a bootable USB:
   - On Windows: Use Rufus or Etcher
   - On Mac: Use Etcher
   - On Linux: Use `dd` or Etcher
3. Boot from the USB drive (you might need to press F12 during startup to select boot device)
4. Once booted into NixOS live environment:
   ```bash
   # Connect to internet (if using WiFi)
   sudo systemctl start wpa_supplicant
   wpa_cli
   > add_network
   > set_network 0 ssid "your_wifi_name"
   > set_network 0 psk "your_wifi_password"
   > enable_network 0
   > quit

   # Download our setup script
   curl -L https://raw.githubusercontent.com/[your-repo]/main/setup-cluster-node.sh -o setup.sh
   chmod +x setup.sh

   # Run the installation
   # For manager node:
   sudo ./setup.sh --role manager
   # For worker node:
   sudo ./setup.sh --role worker
   ```

5. Review the generated configuration and run:
   ```bash
   sudo nixos-install
   ```

6. When prompted, set a root password
7. Reboot the computer

### 4. Post-Installation Setup

On the manager node:

1. Login as root using the password you set
2. Create the shared directories:
   ```bash
   mkdir -p /nfs/project
   mkdir -p /nfs/scratch
   chmod 777 /nfs/project /nfs/scratch
   ```

On each worker node:

1. Login as root
2. Verify connection to manager:
   ```bash
   ping node0
   ```

3. Test NFS mounts:
   ```bash
   ls /common/project
   ls /common/scratch
   ```

### 5. Testing the Cluster

1. Login to any node as mpiuser
2. Create a test file:
   ```bash
   # Create a simple MPI program
   cat > /common/project/hello.c << EOF
   #include <mpi.h>
   #include <stdio.h>

   int main(int argc, char** argv) {
       MPI_Init(&argc, &argv);

       int world_size;
       MPI_Comm_size(MPI_COMM_WORLD, &world_size);

       int world_rank;
       MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

       char processor_name[MPI_MAX_PROCESSOR_NAME];
       int name_len;
       MPI_Get_processor_name(processor_name, &name_len);

       printf("Hello from processor %s, rank %d out of %d processors\n",
              processor_name, world_rank, world_size);

       MPI_Finalize();
       return 0;
   }
   EOF

   # Compile
   mpicc /common/project/hello.c -o /common/project/hello

   # Run on both nodes
   mpirun -np 2 --host node0,node1 /common/project/hello
   ```

## Customizing Your Setup

### Network Configuration

If you need to change network addresses, edit `modules/nixos/networking.nix` and modify:

```nix
networking = {
  hosts = {
    "10.0.0.1" = ["node0"];
    "10.0.0.2" = ["node1"];
    # Add more nodes here if needed
  };
};
```

### Adding More Worker Nodes

1. Follow the installation steps for worker nodes
2. Add the new node's IP and hostname to `modules/nixos/networking.nix`
3. Update the hosts file on all nodes

### CUDA Configuration

CUDA is pre-configured, but you can modify CUDA-related settings in `modules/nixos/nvidia.nix`.

## Common Issues and Solutions

### Network Issues

If nodes can't see each other:
1. Check physical connections
2. Verify IP addresses match configuration
3. Test with `ping`
4. Check network interface names match your hardware

### NFS Issues

If NFS mounts fail:
1. On manager node:
   ```bash
   systemctl status nfs-server
   ```
2. On worker nodes:
   ```bash
   systemctl status nfs-client.target
   ```

### MPI Issues

If MPI jobs fail:
1. Verify SSH works between nodes:
   ```bash
   # From node0
   ssh node1 hostname
   ```
2. Check MPI installation:
   ```bash
   mpirun --version
   ```

## Getting Help

If you encounter issues:
1. Check the [NixOS Manual](https://nixos.org/manual/nixos/stable/)
2. Ask your instructor/TA
3. Create an issue in this repository

## Understanding the Code

The configuration is organized into:
- `flake.nix`: Main configuration file
- `modules/`: Shared configurations
  - `nixos/`: System configurations
  - `home-manager/`: User configurations
- `hosts/`: Machine-specific configurations
  - `manager/`: Manager node config
  - `worker/`: Worker node config

## Contributing

Found a bug or want to improve something? Please:
1. Fork the repository
2. Create a branch
3. Make your changes
4. Submit a pull request
