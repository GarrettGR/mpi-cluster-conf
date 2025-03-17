{ node, nodes, filesystemConfig }:

{ config, pkgs, lib, ... }:

let
  isController = node.isController or false;
  filesystemType = filesystemConfig.type or "nfs";
  
  # Implementation for various filesystem types
  filesystemImplementations = {
    # NFS filesystem configuration
    nfs = {
      server = lib.mkIf isController {
        services.nfs.server = {
          enable = true;
          exports = lib.concatMapStringsSep "\n" 
            (mount: "${mount.exportPath} ${filesystemConfig.exportOptions or "*(rw,sync,no_subtree_check,no_root_squash)"}") 
            filesystemConfig.sharedMounts;
        };
        
        # Create the required directories on the controller
        system.activationScripts.createNfsExports = ''
          ${lib.concatMapStringsSep "\n" 
            (mount: "mkdir -p ${mount.exportPath} && chmod 755 ${mount.exportPath}") 
            filesystemConfig.sharedMounts}
        '';
      };
      
      client = lib.mkIf (!isController) {
        fileSystems = lib.listToAttrs (map 
          (mount: {
            name = mount.mountPoint;
            value = {
              device = "${nodes.${filesystemConfig.serverNode or "controller"}.ip}:${mount.exportPath}";
              fsType = "nfs";
              options = mount.options or [ "noatime" "nodiratime" "x-systemd.automount" "x-systemd.idle-timeout=600" ];
            };
          }) 
          filesystemConfig.sharedMounts);
      };
    };
    
    # BeeGFS filesystem configuration
    beegfs = {
      server = lib.mkIf (lib.elem node.hostname (filesystemConfig.serverNodes or [])) {
        services.beegfs = {
          enable = true;
          
          # Management service (on primary server only)
          mgmtd = lib.mkIf (node.hostname == lib.elemAt (filesystemConfig.serverNodes or [""]) 0) {
            enable = true;
            storePath = "/var/lib/beegfs/mgmtd";
          };
          
          # Metadata service (on primary server only)
          meta = lib.mkIf (node.hostname == lib.elemAt (filesystemConfig.serverNodes or [""]) 0) {
            enable = true;
            storePath = "/var/lib/beegfs/meta";
            mgmtdHost = node.hostname;
          };
          
          # Storage services
          storage = {
            enable = true;
            storePath = "/var/lib/beegfs/storage"; # FIX: path???
            mgmtdHost = lib.elemAt (filesystemConfig.serverNodes or [""]) 0;
          };
          
          # Admon service (on primary server only)
          admon = lib.mkIf (node.hostname == lib.elemAt (filesystemConfig.serverNodes or [""]) 0) {
            enable = true;
            dbPath = "/var/lib/beegfs/admon";
            mgmtdHost = node.hostname;
          };
          
          # Helper service
          helperd.enable = true;
          
          # Use rdma if available
          connUseRDMA = filesystemConfig.useRDMA or false;
          
          # Additional settings
          extraConfig = filesystemConfig.extraConfig or "";
        };
        
        # Create BeeGFS storage directories
        system.activationScripts.createBeegfsDirs = ''
          mkdir -p /var/lib/beegfs/meta
          mkdir -p /var/lib/beegfs/storage
          mkdir -p /var/lib/beegfs/mgmtd
          mkdir -p /var/lib/beegfs/admon
          chmod -R 755 /var/lib/beegfs
        '';
        
        # Configure optimized storage for BeeGFS
        boot.kernelModules = [ "configfs" ];
        boot.kernelParams = [ "cgroup_enable=memory" "swapaccount=1" ];
      };
      
      client = {
        services.beegfs.client = {
          enable = true;
          mgmtdHost = lib.elemAt (filesystemConfig.serverNodes or [""]) 0;
          
          # Mount points configuration
          mounts = lib.listToAttrs (map 
            (mount: {
              name = mount.mountPoint;
              value = {
                cfg = {
                  connMaxInternodeNum = 8;
                  connRDMA = filesystemConfig.useRDMA or false;
                  tuneNumWorkers = 16;
                };
                mountPoint = mount.mountPoint;
                targetStripePattern = "raid0";
                targetNumTargets = filesystemConfig.numTargets or 4;
              };
            }) 
            filesystemConfig.sharedMounts);
            
          # Additional client settings
          extraConfig = filesystemConfig.clientExtraConfig or "";
        };
      };
    };
    
    # Lustre storage configuration
    lustre = {
      server = lib.mkIf (lib.elem node.hostname (filesystemConfig.serverNodes or [])) {
        # Build Lustre kernel module
        boot.kernelPackages = lib.mkForce pkgs.linuxPackages_latest;
        boot.extraModulePackages = with config.boot.kernelPackages; [ lustre ];
        boot.kernelModules = [ "lustre" ];
        
        # Configure Lustre server
        systemd.services.lustre-server = {
          description = "Lustre Server";
          wantedBy = [ "multi-user.target" ];
          after = [ "network.target" ];
          script = ''
            # Example Lustre server setup script
            # This is a placeholder - real Lustre setup is more complex
            ${pkgs.kmod}/bin/modprobe lustre
            mkdir -p /mnt/mdt /mnt/ost
            # In reality, you would use mkfs.lustre and mount specific Lustre filesystems here
          '';
          serviceConfig = {
            Type = "oneshot";
            RemainAfterExit = true;
          };
        };
      };
      
      client = {
        # Build Lustre kernel module
        boot.kernelPackages = lib.mkForce pkgs.linuxPackages_latest;
        boot.extraModulePackages = with config.boot.kernelPackages; [ lustre ];
        boot.kernelModules = [ "lustre" ];
        
        # Lustre client mounts
        fileSystems = lib.listToAttrs (map 
          (mount: {
            name = mount.mountPoint;
            value = {
              device = "${filesystemConfig.mgsDomain or "mgs@o2ib"}:/${mount.lustreFilesystem or "lustre"}";
              fsType = "lustre";
              options = mount.options or [ "noatime" "user_xattr" "flock" ];
            };
          }) 
          filesystemConfig.sharedMounts);
      };
    };
    
    # Simple local storage configuration (for testing)
    local = {
      server = lib.mkIf isController {
        system.activationScripts.createLocalSharedDirs = ''
          ${lib.concatMapStringsSep "\n" 
            (mount: "mkdir -p ${mount.exportPath} && chmod 777 ${mount.exportPath}") 
            filesystemConfig.sharedMounts}
        '';
      };
      
      client = {};
    };
  };
  
  # Get the right implementation based on storage type
  implementation = filesystemImplementations.${filesystemType} or filesystemImplementations.nfs;
  
in lib.mkMerge [
  # Apply server or client config based on node type
  (implementation.server or {})
  (implementation.client or {})
  
  # Common storage configuration for all nodes
  {
    # Add file system optimization settings
    boot.kernel.sysctl = {
      # Increase file max for HPC workloads
      "fs.file-max" = 10000000;
      # Adjust inode cache
      "fs.inotify.max_user_watches" = 524288;
      # Adjust node cache settings
      "vm.dirty_background_ratio" = 5;
      "vm.dirty_ratio" = 10;
      "vm.swappiness" = 10;
    };
    
    # Add tools for storage management
    environment.systemPackages = with pkgs; [
      lsof
      iotop
      sysstat
      hdparm
      smartmontools
      nfs-utils
      xfsprogs
    ] ++ (if filesystemType == "beegfs" then [ beegfs-client ] else [])
      ++ (if filesystemType == "lustre" then [ lfs ] else []);
      
    # Create common mount points
    system.activationScripts.createMountPoints = ''
      ${lib.concatMapStringsSep "\n" 
        (mount: "mkdir -p ${mount.mountPoint}") 
        filesystemConfig.sharedMounts}
    '';
    
    # Add high performance file system modules
    boot.supportedFilesystems = [ "xfs" "nfs" "nfs4" ];
    boot.extraModprobeConfig = ''
      options nfs fscache=yes
      options nfs acregmin=30 acregmax=30 acdirmin=30 acdirmax=30
    '';
    
    # Scratch directory management
    systemd.services.manage-scratch = lib.mkIf (filesystemConfig.enableScratchCleanup or false) {
      description = "Clean up old files in scratch directory";
      startAt = filesystemConfig.scratchCleanupSchedule or "weekly";
      script = ''
        ${pkgs.findutils}/bin/find ${filesystemConfig.scratchDir or "/scratch"} -type f -atime +${toString (filesystemConfig.scratchMaxAge or 30)} -delete
      '';
      serviceConfig = {
        Type = "oneshot";
        ExecStartPre = "${pkgs.coreutils}/bin/mkdir -p ${filesystemConfig.scratchDir or "/scratch"}";
      };
    };
    
    # Storage performance monitoring
    services.telegraf.inputs.diskio = lib.mkIf config.services.telegraf.enable {
      interval = "30s";
      devices = [ "*" ];
      skip_serial_number = true;
    };
  }
]
