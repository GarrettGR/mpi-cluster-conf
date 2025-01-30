{config, pkgs, ...}: {
  imports = [
    <nixpkgs/nixos/modules/installer/cd-dvd/installation-cd-minimal.nix>
  ];

  environment.systemPackages = with pkgs; [
    git
    vim
    wget
    curl
  ];

  system.activationScripts.installClusterConfig = ''
    mkdir -p /etc/nixos/cluster-config
    cp -r ${./..} /etc/nixos/cluster-config/
  '';

  system.activationScripts.clusterSetup = ''
    cat > /root/setup-cluster.sh << 'EOF'
    #!/usr/bin/env bash
    echo "Welcome to NixOS MPI Cluster Installer"
    echo "1. Manager Node"
    echo "2. Worker Node"
    read -p "Select node type (1/2): " node_type
    
    case $node_type in
      1) role="manager" ;;
      2) role="worker" ;;
      *) echo "Invalid selection"; exit 1 ;;
    esac

    /etc/nixos/cluster-config/setup-cluster-node.sh --role "$role"
    EOF

    chmod +x /root/setup-cluster.sh
  '';
}
