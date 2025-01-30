{config, lib, ...}: 
let 
  inherit (lib) mkOption types;
in {
  options = {
    cluster = {
      networking = {
        hostName = mkOption {
          type = types.str;
          description = "Hostname of the machine";
        };
        ipAddress = mkOption {
          type = types.str;
          description = "Primary IP address for the cluster network";
        };
        publicAddress = mkOption {
          type = types.str;
          description = "Public-facing IP address";
        };
        interface = mkOption {
          type = types.str;
          description = "Network interface name";
        };
      };
    };
  };

  config = {
    networking = {
      networkmanager.enable = true;
      firewall.enable = false;
      nameservers = [ "8.8.8.8" ];
      defaultGateway = "192.168.1.1";
      
      hostName = config.cluster.networking.hostName;
      
      hosts = {
        "10.0.0.1" = ["node0"];
        "10.0.0.2" = ["node1"];
      };

      interfaces.${config.cluster.networking.interface}.ipv4.addresses = [
        { 
          address = config.cluster.networking.ipAddress;
          prefixLength = 24;
        }
        {
          address = config.cluster.networking.publicAddress;
          prefixLength = 24;
        }
      ];
    };
  };
}
