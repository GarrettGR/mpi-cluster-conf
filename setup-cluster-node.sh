#!/usr/bin/env bash

set -euo pipefail

ROLE=""
INTERFACE=""

print_usage() {
  echo "Usage: $0 --role [manager|worker] [--interface NETWORK_INTERFACE]"
  echo "Example: $0 --role manager --interface enp0s13f0u2"
}

while [[ $# -gt 0 ]]; do
  case $1 in
    --role)
      ROLE="$2"
      shift 2
    ;;
    --interface)
      INTERFACE="$2"
      shift 2
    ;;
    *)
      echo "Unknown option: $1"
      print_usage
      exit 1
    ;;
  esac
done

if [[ ! "$ROLE" =~ ^(manager|worker)$ ]]; then
  echo "Error: Role must be either 'manager' or 'worker'"
  print_usage
  exit 1
fi

nixos-generate-config --no-filesystems --show-hardware-config > /tmp/hardware-scan.nix

ROOT_UUID=$(lsblk -o UUID,MOUNTPOINT -n | grep '/ $' | awk '{print $1}')
BOOT_UUID=$(lsblk -o UUID,MOUNTPOINT -n | grep '/boot$' | awk '{print $1}')
SWAP_UUID=$(lsblk -o UUID,TYPE -n | grep 'swap' | awk '{print $1}')

if [[ -z "$INTERFACE" ]]; then
  INTERFACE=$(ip -o link show | grep -v lo | head -n1 | awk -F': ' '{print $2}')
  echo "Auto-detected network interface: $INTERFACE"
fi

cat > /etc/nixos/hardware-configuration.nix << EOF
{ config, lib, pkgs, ... }:

{
  imports = [ ../modules/nixos/hardware.nix ];

  cluster.hardware = {
    rootDevice = "$ROOT_UUID";
    bootDevice = "$BOOT_UUID";
    swapDevice = "$SWAP_UUID";
    availableKernelModules = [
      $(grep "boot.initrd.availableKernelModules" /tmp/hardware-scan.nix | sed 's/.*= \[ //' | sed 's/ \];//')
    ];
  };

  cluster.networking = {
    hostName = "node0";
    ipAddress = "10.0.0.1";
    publicAddress = "192.168.1.201";
    interface = "$INTERFACE";
  };
}
EOF

if [[ ! -d /etc/nixos/cluster-config ]]; then
  git clone https://github.com/garrettgr/nixos-mpi-cluster /etc/nixos/cluster-config
fi

if [[ "$ROLE" == "manager" ]]; then
  ln -sf /etc/nixos/cluster-config/hosts/manager /etc/nixos/configuration.nix
else
  ln -sf /etc/nixos/cluster-config/hosts/worker /etc/nixos/configuration.nix
fi

echo "Configuration generated for $ROLE node"
echo "Review the configuration and run: sudo nixos-rebuild switch"
