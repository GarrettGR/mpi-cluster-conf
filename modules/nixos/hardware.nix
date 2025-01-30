{config, lib, pkgs, modulesPath, ...}:
let
  inherit (lib) mkOption types;
in {
  options = {
    cluster = {
      hardware = {
        rootDevice = mkOption {
          type = types.str;
          description = "UUID of the root filesystem";
        };
        bootDevice = mkOption {
          type = types.str;
          description = "UUID of the boot filesystem";
        };
        swapDevice = mkOption {
          type = types.str;
          description = "UUID of the swap device";
        };
        availableKernelModules = mkOption {
          type = types.listOf types.str;
          default = [
            "xhci_pci"
            "ahci"
            "nvme"
            "usb_storage"
            "sd_mod"
            "rtsx_pci_sdmmc"
          ];
          description = "Available kernel modules for the hardware";
        };
      };
    };
  };

  config = {
    imports = [
      (modulesPath + "/installer/scan/not-detected.nix")
    ];

    boot.initrd.availableKernelModules = config.cluster.hardware.availableKernelModules;
    boot.initrd.kernelModules = [ ];
    boot.kernelModules = [ "kvm-intel" ];
    boot.extraModulePackages = [ ];

    fileSystems."/" = {
      device = "/dev/disk/by-uuid/${config.cluster.hardware.rootDevice}";
      fsType = "ext4";
    };

    fileSystems."/boot" = {
      device = "/dev/disk/by-uuid/${config.cluster.hardware.bootDevice}";
      fsType = "vfat";
      options = [ "fmask=0077" "dmask=0077" ];
    };

    swapDevices = [{
      device = "/dev/disk/by-uuid/${config.cluster.hardware.swapDevice}";
    }];

    nixpkgs.hostPlatform = lib.mkDefault "x86_64-linux";
    hardware.cpu.intel.updateMicrocode = lib.mkDefault config.hardware.enableRedistributableFirmware;
  };
}
