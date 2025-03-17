# Modular NixOS HPC Cluster Flake

A highly modular NixOS flake for creating High Performance Computing (HPC) clusters with SLURM, OpenMPI, CUDA, and flexible storage options.

## Features

- **Modular design**: Easily swap components like storage systems
- **CUDA-aware MPI**: Optimized for GPU computing with CUDA-aware OpenMPI
- **Storage flexibility**: Support for NFS, BeeGFS, and Lustre
- **User management**: Home Manager integration for user environment configuration
- **Scalable**: Works with clusters of any size, from 2 nodes to hundreds
- **Monitoring**: Prometheus, Grafana, and GPU metrics integrated

## Repository Structure

```
.
├── flake.nix              # Main flake entry point
├── lib/                   # Utility functions
│   └── default.nix        # Common utility functions
├── modules/               # Modular configuration components
│   ├── base/              # Base system configuration
│   ├── hpc/               # HPC software and CUDA configuration
│   ├── monitoring/        # Monitoring and metrics
│   ├── networking/        # Network configuration
│   ├── slurm/             # SLURM workload manager
│   ├── storage/           # Storage systems (NFS, BeeGFS, etc.)
│   └── users/             # User management with Home Manager
└── examples/              # Example configurations
```

## Quick Start

### Option 1: Use directly from GitHub

Create a configuration file for your cluster (e.g., `my-cluster.nix`):

```nix
let
  clusterFlake = import (fetchTarball "https://github.com/yourusername/nixos-hpc-cluster/archive/main.tar.gz");
in clusterFlake.mkClusterFlake {
  # Basic cluster configuration
  controllerHostname = "controller";
  controllerIP = "10.0.0.1";
  
  # Define worker nodes
  workerNodes = [
    { hostname = "worker1"; ip = "10.0.0.11"; hasGPU = true; cpus = 64; memoryMB = 196608; gpus = 4; }
    { hostname = "worker2"; ip = "10.0.0.12"; hasGPU = true; cpus = 64; memoryMB = 196608; gpus = 4; }
  ];
  
  # Storage configuration (defaults to NFS)
  storageConfig = {
    type = "nfs";  # Options: "nfs", "beegfs", "lustre", "local"
    sharedMounts = [
      { mountPoint = "/home"; exportPath = "/home"; }
      { mountPoint = "/opt/shared"; exportPath = "/opt/shared"; }
    ];
  };
  
  # SSH keys for access
  sshKeys = [
    "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAA... user@example"
  ];
  
  # Optional user configurations
  users = [
    {
      name = "researcher";
      groups = [ "users" ];
      nodes = "all";
      homeConfig = { pkgs, ... }: {
        home.packages = with pkgs; [ python3 git ];
      };
    }
  ];
}
```

Build and deploy:

```bash
# For the controller
nixos-rebuild switch --flake ./my-cluster.nix#controller

# For worker nodes
nixos-rebuild switch --flake ./my-cluster.nix#worker1
```

### Option 2: Include as a flake input

In your `flake.nix`:

```nix
{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    nixos-hpc-cluster.url = "github:yourusername/nixos-hpc-cluster";
  };

  outputs = { self, nixpkgs, nixos-hpc-cluster, ... }: {
    nixosConfigurations = nixos-hpc-cluster.mkClusterFlake {
      controllerHostname = "controller";
      controllerIP = "10.0.0.1";
      workerNodes = [
        { hostname = "worker1"; ip = "10.0.0.11"; hasGPU = true; cpus = 64; memoryMB = 196608; gpus = 4; }
      ];
      sshKeys = [ "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAA... user@example" ];
    };
  };
}
```

## Module Configuration Options

Each module can be customized independently:

### Storage Options

```nix
storageConfig = {
  # NFS configuration
  type = "nfs";
  sharedMounts = [
    { mountPoint = "/home"; exportPath = "/home"; }
    { mountPoint = "/opt/shared"; exportPath = "/opt/shared"; }
  ];
  exportOptions = "*(rw,sync,no_subtree_check,no_root_squash)";
};

# OR

storageConfig = {
  # BeeGFS configuration
  type = "beegfs";
  sharedMounts = [
    { mountPoint = "/home"; beegfsDataTarget = "home"; }
    { mountPoint = "/scratch"; beegfsDataTarget = "scratch"; }
  ];
  serverNodes = [ "worker1" ];  # Nodes that will run BeeGFS services
  useRDMA = true;               # Enable RDMA transport
  numTargets = 4;               # Number of storage targets
};

# OR

storageConfig = {
  # Lustre configuration
  type = "lustre";
  sharedMounts = [
    { mountPoint = "/lustre"; lustreFilesystem = "lustre"; }
  ];
  serverNodes = [ "worker1" "worker2" ];
  mgsDomain = "mgs@o2ib";
};
```

### Networking Options

```nix
networkConfig = {
  domain = "cluster.local";
  subnet = "10.0.0";
  netmask = "255.255.255.0";
  defaultGateway = "10.0.0.1";
  
  # Optional
  enableFirewall = true;
  extraAllowedTCPPorts = [ 8080 9090 ];
  
  # InfiniBand support
  enableInfiniband = true;
  infinibandHardware = "mlx5_0";
  
  # Mellanox support
  enableMellanox = true;
};
```

### User Management with Home Manager

```nix
users = [
  # Admin user on all nodes
  {
    name = "admin";
    groups = [ "wheel" "networkmanager" ];
    nodes = "all";  # Options: "all", "controller", "workers", or list of hostnames
    homeConfig = { pkgs, ... }: {
      home.packages = with pkgs; [ htop tmux vim ];
    };
  },
  
  # Researcher only on compute nodes
  {
    name = "researcher";
    nodes = "workers";
    homeConfig = { pkgs, ... }: {
      home.packages = with pkgs; [ python3 tensorflow jupyter ];
    };
  }
];
```

## Extending and Customizing

### Adding Custom Modules

You can add custom modules specific to nodes or applied to all nodes:

```nix
extraNodeModules = {
  controller = { config, pkgs, ... }: {
    # Configuration only for controller
    services.custom-service.enable = true;
  };
  
  worker1 = { config, pkgs, ... }: {
    # Configuration only for worker1
    hardware.custom-hardware.enable = true;
  };
};

extraCommonModules = [
  # Applied to all nodes
  ({ config, pkgs, ... }: {
    services.tailscale.enable = true;
  })
];
```

### Replacing Components

The modular design makes it easy to replace components. For example, to switch from NFS to BeeGFS:

```nix
storageConfig = {
  type = "beegfs";
  sharedMounts = [
    { mountPoint = "/home"; beegfsDataTarget = "home"; }
    { mountPoint = "/scratch"; beegfsDataTarget = "scratch"; }
  ];
  serverNodes = [ "worker1" ];
};
```

## Adding New Storage Backends

The modular design makes it easy to add new storage backends. To add a new storage type:

1. Edit `modules/storage/default.nix`
2. Add your new implementation to the `storageImplementations` map:

```nix
storageImplementations = {
  # Existing implementations...
  
  # Your new storage system
  myCustomStorage = {
    server = lib.mkIf isServer {
      # Server-side configuration
      services.myCustomStorage.server = {
        enable = true;
        # Other options...
      };
    };
    
    client = lib.mkIf (!isServer) {
      # Client-side configuration
      services.myCustomStorage.client = {
        enable = true;
        # Other options...
      };
    };
  };
};
```

3. Use it in your cluster configuration:

```nix
storageConfig = {
  type = "myCustomStorage";
  # Your custom options...
};
```

## Post-Deployment Steps

After deploying the configuration to all nodes:

1. **Generate a munge key** for SLURM authentication:
   ```bash
   # On the controller
   nix run ./my-cluster.nix -- generate-munge-key
   # Copy the generated munge.key to /etc/munge/munge.key on all nodes
   ```

2. **Restart munge and SLURM** on all nodes:
   ```bash
   systemctl restart munge
   systemctl restart slurmd  # On workers
   systemctl restart slurmctld  # On controller
   ```

3. **Verify the cluster** is working:
   ```bash
   # On the controller
   sinfo
   srun -N2 hostname
   ```

## Testing CUDA-aware MPI

The flake includes a utility to generate a CUDA-aware MPI test program:

```bash
# Generate the test program
nix run ./my-cluster.nix -- test-cuda-mpi

# Compile and run it
mpicc -o cuda_mpi_test cuda_mpi_test.c -lcudart
mpirun -np 4 --hostfile /etc/openmpi-hostfile ./cuda_mpi_test
```

## Monitoring

The monitoring module sets up:

1. **Prometheus** on the controller node for metrics collection
2. **Grafana** for visualization (accessible at http://controller:3000)
3. **Node Exporter** on all nodes for system metrics
4. **NVIDIA GPU Exporter** on GPU nodes

Default credentials for Grafana:
- Username: `admin`
- Password: `admin`

## Known Limitations

- Lustre support is somewhat limited as it requires external kernel modules
- RDMA support requires specific hardware and may need additional configuration
- The cluster assumes a single controller node (no HA configuration yet)

## Contributing

Contributions are welcome! Areas that would particularly benefit from contributions:

1. Support for high-availability SLURM configuration
2. Additional storage backends (CephFS, GlusterFS, etc.)
3. Integration with cloud providers for dynamic scaling
4. Advanced networking configurations

Please submit pull requests or open issues on GitHub.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
