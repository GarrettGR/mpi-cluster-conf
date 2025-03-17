{ nixpkgs }:

let
  pkgsLib = nixpkgs.lib;
in {
  # Generate nodes map from controller and worker configs
  generateNodes = { controllerHostname, controllerIP, workerNodes }:
    pkgsLib.listToAttrs (
      # Controller node
      [{
        name = controllerHostname;
        value = {
          hostname = controllerHostname;
          ip = controllerIP;
          isController = true;
          hasGPU = false;
          systemConfig = { config, lib, ... }: {
            networking = {
              hostName = controllerHostname;
              interfaces.enp0s3.ipv4.addresses = [{
                address = controllerIP;
                prefixLength = 24;
              }];
            };
          };
        };
      }] ++
      # Worker nodes
      (map (node: {
        name = node.hostname;
        value = {
          hostname = node.hostname;
          ip = node.ip;
          isController = false;
          hasGPU = node.hasGPU or true;
          cpus = node.cpus or 64;
          memoryMB = node.memoryMB or 196608;
          gpus = node.gpus or 4;
          systemConfig = { config, lib, ... }: {
            networking = {
              hostName = node.hostname;
              interfaces.enp0s3.ipv4.addresses = [{
                address = node.ip;
                prefixLength = 24;
              }];
            };
          };
        };
      }) workerNodes)
    );

  # Generate a hosts file from cluster nodes
  generateHostsEntries = { nodes, networkConfig }:
    pkgsLib.concatMapStrings
      (node: "${node.ip} ${node.hostname}.${networkConfig.domain} ${node.hostname}\n")
      (pkgsLib.attrValues nodes);
      
  # Generate SLURM node configuration
  generateSlurmNodeConfig = nodes:
    pkgsLib.concatMapStrings
      (node: "NodeName=${node.hostname} NodeAddr=${node.ip} CPUs=${toString node.cpus} RealMemory=${toString node.memoryMB} ${if node.hasGPU then "Gres=gpu:${toString node.gpus}" else ""} State=UNKNOWN\n")
      (pkgsLib.filter (node: !node.isController) (pkgsLib.attrValues nodes));
        
  # Generate SLURM GRES config for GPU nodes
  generateSlurmGresConfig = nodes:
    pkgsLib.concatMapStrings
      (node: node.hasGPU ? "NodeName=${node.hostname} Name=gpu File=/dev/nvidia[0-${toString (node.gpus - 1)}] Cores=0-15,32-47:16-31,48-63:0-31:32-63\n" : "")
      (pkgsLib.filter (node: !node.isController) (pkgsLib.attrValues nodes));
        
  # Generate worker node list for SLURM partition
  generateWorkerList = nodes:
    let 
      workerNodes = pkgsLib.filter (node: !node.isController) (pkgsLib.attrValues nodes);
      nodeNames = map (node: node.hostname) workerNodes;
    in 
      if nodeNames == [] then "NONE" 
      else if builtins.length nodeNames == 1 then builtins.elemAt nodeNames 0
      else "${builtins.elemAt nodeNames 0}-${builtins.elemAt nodeNames (builtins.length nodeNames - 1)}";
  
  # Custom function to deep merge attrsets
  deepMerge = a: b:
    let
      mergeAttrs = name: values:
        if pkgsLib.isAttrs values.a && pkgsLib.isAttrs values.b
        then deepMerge values.a values.b
        else values.b;
    in pkgsLib.zipAttrsWith mergeAttrs { a = a; b = b; };
    
  # Function to check if a node is in a list of hostnames
  isNodeInList = node: list:
    if list == null then true
    else if list == "all" then true
    else if list == "controller" then node.isController
    else if list == "workers" then !node.isController
    else if pkgsLib.isList list then pkgsLib.elem node.hostname list
    else false;
}
