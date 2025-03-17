{ node, nodes }:

{ config, pkgs, lib, ... }:

let
  isController = node.isController or false;
  
  # SLURM node configuration lines (one per worker node)
  slurmNodeConfig = lib.concatMapStrings
    (n: "NodeName=${n.hostname} NodeAddr=${n.ip} CPUs=${toString (n.cpus or 64)} RealMemory=${toString (n.memoryMB or 196608)} ${if (n.hasGPU or false) then "Gres=gpu:${toString (n.gpus or 4)}" else ""} State=UNKNOWN\n")
    (lib.filter (n: !(n.isController or false)) (lib.attrValues nodes));
  
  # SLURM GRES config for all GPU nodes
  slurmGresConfig = lib.concatMapStrings
    (n: (n.hasGPU or false) ? "NodeName=${n.hostname} Name=gpu File=/dev/nvidia[0-${toString ((n.gpus or 4) - 1)}] Cores=0-15,32-47:16-31,48-63:0-31:32-63\n" : "")
    (lib.filter (n: !(n.isController or false)) (lib.attrValues nodes));
  
  # Generate worker node list for SLURM partition
  workerList = let 
    workerNodes = lib.filter (n: !(n.isController or false)) (lib.attrValues nodes);
    nodeNames = map (n: n.hostname) workerNodes;
  in 
    if nodeNames == [] then "NONE" 
    else if builtins.length nodeNames == 1 then builtins.elemAt nodeNames 0
    else "${builtins.elemAt nodeNames 0}-${builtins.elemAt nodeNames (builtins.length nodeNames - 1)}";
in {
  # SLURM packages
  environment.systemPackages = with pkgs; [
    slurm
    pmix  # Process Management Interface for Exascale
  ];

  # SLURM configuration
  services.slurm = {
    enable = true;
    enableStools = true;
    includeSchedMD = true;
    
    # SLURM configuration file
    configFile = pkgs.writeText "slurm.conf" ''
      # Global configuration
      ClusterName=hpc-cluster
      SlurmctldHost=${lib.findSingle (n: n.isController) (builtins.elemAt (lib.attrValues nodes) 0) (builtins.elemAt (lib.attrValues nodes) 0) (lib.attrValues nodes)}.hostname(${lib.findSingle (n: n.isController) (builtins.elemAt (lib.attrValues nodes) 0) (builtins.elemAt (lib.attrValues nodes) 0) (lib.attrValues nodes)}.ip)
      SlurmUser=slurm
      SlurmdUser=root
      SlurmctldPort=6817
      SlurmdPort=6818
      AuthType=auth/munge
      StateSaveLocation=/var/spool/slurm/ctld
      SlurmdSpoolDir=/var/spool/slurm/d
      SwitchType=switch/none
      MpiDefault=pmix
      SlurmctldPidFile=/var/run/slurm/slurmctld.pid
      SlurmdPidFile=/var/run/slurm/slurmd.pid
      ProctrackType=proctrack/cgroup
      ReturnToService=1
      TaskPlugin=task/affinity,task/cgroup
      SchedulerType=sched/backfill
      SelectType=select/cons_tres
      SelectTypeParameters=CR_CPU_Memory
      AccountingStorageType=accounting_storage/none
      JobAcctGatherType=jobacct_gather/linux
      JobAcctGatherFrequency=30
      SlurmctldLogFile=/var/log/slurm/slurmctld.log
      SlurmdLogFile=/var/log/slurm/slurmd.log
      SlurmctldDebug=info
      SlurmdDebug=info
      
      # GRES (Generic Resource) Configuration for GPUs
      GresTypes=gpu
      
      # Node configurations
      ${slurmNodeConfig}
      
      # Partition configuration
      PartitionName=main Nodes=${workerList} Default=YES MaxTime=INFINITE State=UP
    '';
    
    # GPU GRES configuration for SLURM
    gresConfFile = pkgs.writeText "gres.conf" ''
      # GPU configuration for SLURM
      ${slurmGresConfig}
    '';

    # SLURM cgroup configuration
    cgroupConfFile = pkgs.writeText "cgroup.conf" ''
      CgroupMountpoint=/sys/fs/cgroup
      CgroupAutomount=yes
      CgroupReleaseAgentDir="/etc/slurm/cgroup"
      AllowedDevicesFile="/etc/slurm/cgroup_allowed_devices_file.conf"
      ConstrainDevices=yes
      TaskAffinity=yes
      ConstrainRAMSpace=yes
      ConstrainSwapSpace=yes
      ConstrainCores=yes
      AllowedKmemSpace=16777216
      MemorySwappiness=0
      ConstrainKmemSpace=no
    '';
  };

  # Munge authentication service for SLURM
  services.munge = {
    enable = true;
    password = "/etc/munge/munge.key";
  };

  # SLURM controller specific configuration
  systemd.services.slurmctld = lib.mkIf isController {
    wantedBy = [ "multi-user.target" ];
    requires = [ "network-online.target" "munge.service" ];
    after = [ "network-online.target" "munge.service" ];
  };
  
  # SLURM node specific configuration
  systemd.services.slurmd = lib.mkIf (!isController) {
    wantedBy = [ "multi-user.target" ];
    requires = [ "network-online.target" "munge.service" ];
    after = [ "network-online.target" "munge.service" ];
  };

  # Create required directories
  system.activationScripts.slurmDirs = ''
    mkdir -p /var/spool/slurm/d
    mkdir -p /var/spool/slurm/ctld
    mkdir -p /var/log/slurm
    mkdir -p /var/run/slurm
    chown -R slurm:slurm /var/spool/slurm
    chown -R slurm:slurm /var/log/slurm
    chown -R slurm:slurm /var/run/slurm
  '';
  
  # Add SLURM utilities
  environment.systemPackages = lib.mkIf isController (with pkgs; [
    slurm-spank-plugins
    slurmacs  # SLURM Accounting System
  ]);
  
  # Setup SLURM database (optional, for job accounting)
  services.mysql = lib.mkIf (isController && (config.services.slurm.enableAccounting or false)) {
    enable = true;
    package = pkgs.mariadb;
    ensureDatabases = [ "slurm_acct_db" ];
    ensureUsers = [
      {
        name = "slurm";
        ensurePermissions = {
          "slurm_acct_db.*" = "ALL PRIVILEGES";
        };
      }
    ];
  };
  
  # Configure SLURM database daemon if accounting is enabled
  services.slurm.dbdConfig = lib.mkIf (isController && (config.services.slurm.enableAccounting or false)) ''
    AuthType=auth/munge
    DbdHost=${lib.findSingle (n: n.isController) (builtins.elemAt (lib.attrValues nodes) 0) (builtins.elemAt (lib.attrValues nodes) 0) (lib.attrValues nodes)}.hostname
    StorageType=accounting_storage/mysql
    StorageHost=localhost
    StorageUser=slurm
    StoragePass=slurm
    StorageLoc=slurm_acct_db
  '';
  
  # Add helper scripts for SLURM
  environment.systemPackages = with pkgs; [ # NOTE: Do I even want these ???
    (writeShellScriptBin "slurm-status" ''
      #!/bin/bash
      # Show SLURM status and statistics
      echo "=== SLURM Node Status ==="
      sinfo -N -l
      
      echo -e "\n=== SLURM Partition Status ==="
      sinfo
      
      echo -e "\n=== Running Jobs ==="
      squeue
      
      if [ $(squeue | wc -l) -gt 1 ]; then
        echo -e "\n=== Job Details ==="
        for jobid in $(squeue -h -o %A); do
          echo -e "\nJob $jobid:"
          scontrol show job $jobid
        done
      fi
    '')
    
    (writeShellScriptBin "launch-gpu-job" ''
      #!/bin/bash
      # Helper script to launch GPU jobs with proper resource allocation
      # Usage: launch-gpu-job <num_gpus> <command>
      
      if [ $# -lt 2 ]; then
        echo "Usage: launch-gpu-job <num_gpus> <command> [args...]"
        echo "Example: launch-gpu-job 2 python3 train.py"
        exit 1
      fi
      
      NUM_GPUS="$1"
      shift
      COMMAND="$@"
      
      # Launch the job with GPU allocation
      srun --gres=gpu:$NUM_GPUS -n 1 -c 4 $COMMAND
    '')
  ];
}
