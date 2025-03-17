{ node, nodes, networkConfig }:

{ config, pkgs, lib, ... }:

let
  # Generate hosts entries for all nodes
  hostsEntries = lib.concatMapStrings
    (n: "${n.ip} ${n.hostname}.${networkConfig.domain} ${n.hostname}\n")
    (lib.attrValues nodes);
in {
  networking = {
    domain = networkConfig.domain;
    nameservers = networkConfig.nameservers or [ "8.8.8.8" "8.8.4.4" ];
    defaultGateway = networkConfig.defaultGateway or "${networkConfig.subnet}.1";
    
    firewall = {
      enable = networkConfig.enableFirewall or false;
      
      allowedTCPPorts = lib.optional config.services.openssh.enable 22
        ++ networkConfig.extraAllowedTCPPorts or [];
        
      allowedUDPPorts = networkConfig.extraAllowedUDPPorts or [];
      
      # Additional custom rules
      extraCommands = networkConfig.firewallExtraCommands or "";
    };
    
    extraHosts = hostsEntries;
  };
  
  # Set kernel networking parameters optimized for HPC
  boot.kernel.sysctl = {
    # TCP tuning parameters
    "net.ipv4.tcp_rmem" = "4096 87380 16777216";  # TCP read buffer (min default max)
    "net.ipv4.tcp_wmem" = "4096 65536 16777216";  # TCP write buffer (min default max)
    "net.core.rmem_max" = "16777216";             # Maximum TCP read buffer
    "net.core.wmem_max" = "16777216";             # Maximum TCP write buffer
    "net.core.netdev_max_backlog" = "30000";      # Maximum receive queue
    "net.ipv4.tcp_congestion_control" = "cubic";  # Use cubic congestion control
    "net.ipv4.tcp_mtu_probing" = "1";             # Enable MTU probing
    "net.ipv4.tcp_timestamps" = "1";              # Enable TCP timestamps
  } // (networkConfig.extraSysctls or {});
  
  # Network interface tuning
  systemd.services.tune-interfaces = {
    description = "Tune network interfaces for HPC performance";
    after = [ "network.target" ];
    wantedBy = [ "multi-user.target" ];
    script = ''
      # Use maximum 9000 MTU for Jumbo frames if supported
      for iface in $(${pkgs.iproute2}/bin/ip link show | grep -v lo | awk -F: '$0 !~ "lo|vir|docker|veth" {print $2}' | tr -d ' '); do
        ${pkgs.ethtool}/bin/ethtool -K $iface tso on gso on gro on
        # Set rx/tx ring buffer to max
        max_ring=$(${pkgs.ethtool}/bin/ethtool -g $iface | grep -A2 "Pre-set maximums" | tail -1 | awk '{print $1}')
        if [ ! -z "$max_ring" ]; then
          ${pkgs.ethtool}/bin/ethtool -G $iface tx $max_ring rx $max_ring
        fi
        # Try to increase MTU for better performance (may cause issues if network doesn't support jumbo frames) 
        # ${pkgs.iproute2}/bin/ip link set $iface mtu 9000
      done
    '';
    serviceConfig = {
      Type = "oneshot";
      RemainAfterExit = true;
    };
  };
  
  hardware.infiniband = lib.mkIf (networkConfig.enableInfiniband or false) { # (optimal?) InfiniBand support -- if available
    enable = true;
    enableSystemTools = true;
    
    hardware = networkConfig.infinibandHardware or "mlx5_0"; # RDMA (hardware-specific) settings
    cores = networkConfig.infinibandCores or "0-15";
  };
  
  boot.kernelModules = lib.mkIf (networkConfig.enableMellanox or false) [ # Mellanox ConnectX support
    "mlx5_core"
    "mlx5_ib"
    "ib_uverbs"
    "ib_umad"
    "rdma_ucm"
  ];
}
