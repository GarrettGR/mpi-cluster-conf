# Module Development Guide

This guide explains how to develop and extend the modules in the NixOS HPC Cluster flake.

## Module Structure

Each module follows a similar pattern:

```nix
# modules/example/default.nix
{ node, nodes, ... }:  # Input parameters from the main flake

{ config, pkgs, lib, ... }:  # NixOS module parameters

let
  # Local variables and helper functions
  isController = node.isController or false;
  
in {
  # Module configuration
  # ...
}
```

## Creating a New Module

1. Create a new directory under `modules/`:
   ```
   mkdir modules/mymodule
   ```

2. Create a `default.nix` file in your module directory:
   ```nix
   # modules/mymodule/default.nix
   
   { node, nodes, myModuleConfig ? {} }:  # Add your config parameters
   
   { config, pkgs, lib, ... }:
   
   let
     isController = node.isController or false;
     cfg = myModuleConfig;
   in {
     # Your module configuration here
     environment.systemPackages = with pkgs; [
       # Packages to install
     ];
     
     # Services to enable
     services.myservice = {
       enable = true;
       # Other options
     };
     
     # Conditional configurations
     services.controllerOnlyService = lib.mkIf isController {
       enable = true;
     };
   }
   ```

3. Add your module to the imports in the main flake:
   ```nix
   # In flake.nix
   modules = {
     # Existing modules...
     mymodule = import ./modules/mymodule;
   };
   ```

4. Update the `mkSystem` function to include your module:
   ```nix
   mkSystem = name: node:
     nixpkgs.lib.nixosSystem {
       inherit system;
       specialArgs = { inherit pkgs node nodes networkConfig storageConfig myModuleConfig; };
       modules = [
         # Existing modules...
         (modules.mymodule { inherit node nodes myModuleConfig; })
       ];
     };
   ```

5. Add your configuration parameter to `mkClusterFlake`:
   ```nix
   mkClusterFlake = { 
     # Existing parameters...
     myModuleConfig ? {},
   }:
   ```

## Example: Adding a Container Orchestration Module

Let's create a module for container orchestration with Kubernetes or K3s:

1. Create the directory and file:
   ```
   mkdir -p modules/containers
   touch modules/containers/default.nix
   ```

2. Implement the module:
   ```nix
   # modules/containers/default.nix
   
   { node, nodes, containerConfig ? { enable = false; type = "k3s"; } }:
   
   { config, pkgs, lib, ... }:
   
   let
     isController = node.isController or false;
     enable = containerConfig.enable or false;
     type = containerConfig.type or "k3s";
     
     # Implementation for different container types
     containerImplementations = {
       # K3s lightweight Kubernetes
       k3s = {
         server = lib.mkIf isController {
           services.k3s = {
             enable = true;
             role = "server";
             extraFlags = containerConfig.serverExtraFlags or "";
           };
           
           # Open firewall ports
           networking.firewall.allowedTCPPorts = [ 6443 ];
         };
         
         agent = lib.mkIf (!isController) {
           services.k3s = {
             enable = true;
             role = "agent";
             serverAddr = "https://${(lib.findSingle (n: n.isController) (builtins.elemAt (lib.attrValues nodes) 0) (builtins.elemAt (lib.attrValues nodes) 0) (lib.attrValues nodes)).ip}:6443";
             token = containerConfig.token or "/var/lib/rancher/k3s/server/node-token";
             extraFlags = containerConfig.agentExtraFlags or "";
           };
         };
       };
       
       # Full Kubernetes
       kubernetes = {
         server = lib.mkIf isController {
           services.kubernetes = {
             roles = ["master"];
             kubelet.extraOpts = "--node-ip=${node.ip}";
             addons.dns.enable = true;
           };
         };
         
         agent = lib.mkIf (!isController) {
           services.kubernetes = {
             roles = ["node"];
             kubelet.extraOpts = "--node-ip=${node.ip}";
             masterAddress = (lib.findSingle (n: n.isController) (builtins.elemAt (lib.attrValues nodes) 0) (builtins.elemAt (lib.attrValues nodes) 0) (lib.attrValues nodes)).hostname;
           };
         };
       };
     };
     
     # Get the implementation based on container type
     implementation = containerImplementations.${type} or containerImplementations.k3s;
   
   in lib.mkIf enable (lib.mkMerge [
     (implementation.server or {})
     (implementation.agent or {})
     
     # Common container configuration for all nodes
     {
       # Container runtime - containerd
       virtualisation.containerd = {
         enable = true;
         settings = {
           plugins."io.containerd.grpc.v1.cri" = {
             # Add GPU support if needed
             enable_gpu = lib.mkIf (node.hasGPU or false) true;
           };
         };
       };
       
       # Add container tools to the system
       environment.systemPackages = with pkgs; [
         kubectl
         helm
         k9s
       ];
       
       # NVIDIA container runtime for GPU nodes
       virtualisation.docker = lib.mkIf (node.hasGPU or false) {
         enable = true;
         enableNvidia = true;
       };
     }
   ])
   ```

3. Add it to the main flake:
   ```nix
   # In flake.nix
   modules = {
     # Existing modules...
     containers = import ./modules/containers;
   };
   
   mkSystem = name: node:
     nixpkgs.lib.nixosSystem {
       inherit system;
       specialArgs = { inherit pkgs node nodes networkConfig storageConfig containerConfig; };
       modules = [
         # Existing modules...
         (modules.containers { inherit node nodes containerConfig; })
       ];
     };
     
   mkClusterFlake = { 
     # Existing parameters...
     containerConfig ? { enable = false; type = "k3s"; },
   }: 
   ```

4. Use it in a cluster configuration:
   ```nix
   let
     clusterFlake = import ./path/to/flake.nix;
   in clusterFlake.mkClusterFlake {
     # Basic configuration...
     
     # Enable container orchestration with K3s
     containerConfig = {
       enable = true;
       type = "k3s";
       serverExtraFlags = "--no-deploy traefik";
     };
   }
   ```

## Guidelines for Module Development

1. **Use conditional activation** - Always use `lib.mkIf` for conditional parts of your configuration.

2. **Parameterize everything** - Don't hardcode values that users might want to change.

3. **Separate server and client configurations** - Most HPC services have a server and client components.

4. **Document your module** - Add comments explaining options and requirements.

5. **Consider GPU support** - If your module interacts with GPUs, ensure it's properly integrated with CUDA.

6. **Optimize for HPC workloads** - Tune parameters for high performance by default.

7. **Add helper scripts and utilities** - Provide scripts that help users work with your module.

8. **Test on different cluster sizes** - Ensure your module works on both small and large clusters.

## Testing Your Module

1. Create a minimal test configuration:
   ```nix
   let
     clusterFlake = import ./path/to/flake.nix;
   in clusterFlake.mkClusterFlake {
     # Minimal configuration with your module
     myModuleConfig = {
       enable = true;
       # Other options to test
     };
   }
   ```

2. Build and test on a single node:
   ```bash
   nixos-rebuild build --flake ./test-config.nix#controller
   ```

3. Test deployment on multiple nodes if possible.

## Common Module Patterns

### Feature Detection

```nix
# Detect if a feature is available
let
  hasFeature = config.hardware.someFeature.available or false;
in {
  # Enable only if available
  services.myservice = lib.mkIf hasFeature {
    enable = true;
  };
}
```

### Overlay Module Options

```nix
# Override or extend an existing module
{
  # Use mkForce to override a setting
  services.existingModule.setting = lib.mkForce "new-value";
  
  # Use mkBefore/mkAfter to add to a list
  environment.systemPackages = lib.mkBefore [ pkgs.newPackage ];
}
```

### Module Composition

```nix
# Combine multiple sub-configurations
lib.mkMerge [
  # Base configuration
  {
    services.base.enable = true;
  }
  
  # Optional component
  (lib.mkIf cfg.enableComponent {
    services.component.enable = true;
  })
]
```

## Integration with Existing Modules

When your module needs to interact with other modules:

1. Add conditional dependencies:
   ```nix
   {
     # Enable CUDA integration if the node has GPUs
     services.myService.cudaSupport = lib.mkIf (node.hasGPU or false) true;
     
     # Depend on SLURM if enabled
     services.myService.slurmIntegration = lib.mkIf config.services.slurm.enable true;
   }
   ```

2. Expose your services to other modules:
   ```nix
   {
     # Expose a config option that other modules can check
     cluster.services.mymodule.available = true;
     
     # Provide paths that other modules might need
     cluster.services.mymodule.binPath = "${pkgs.mypackage}/bin";
   }
   ```

## Conclusion

By following this guide, you can develop new modules that extend the functionality of the NixOS HPC cluster flake. Remember to maintain the modular design so users can easily customize and swap components as needed.

If you develop a module that might be useful to others, consider submitting it as a pull request to the main repository!
