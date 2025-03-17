# modules/monitoring/default.nix - Monitoring and metrics configuration

{ node, nodes }:

{ config, pkgs, lib, ... }:

let
  isController = node.isController or false;
  hasGPU = node.hasGPU or false;
  
  # Determine which nodes should be monitored
  monitoredNodes = lib.filterAttrs (name: n: true) nodes;
  
  # Controller-specific monitoring configuration
  controllerMonitoring = lib.mkIf isController {
    # Prometheus server configuration
    services.prometheus = {
      enable = true;
      
      # Scrape configurations for all nodes
      scrapeConfigs = [
        {
          job_name = "node";
          static_configs = [{
            targets = lib.mapAttrsToList (name: n: "${n.hostname}:9100") monitoredNodes;
          }];
        }
        {
          job_name = "nvidia";
          static_configs = [{
            targets = lib.mapAttrsToList 
              (name: n: "${n.hostname}:9835") 
              (lib.filterAttrs (name: n: n.hasGPU or false) monitoredNodes);
          }];
        }
        {
          job_name = "slurm";
          static_configs = [{
            targets = [ "${node.hostname}:9100" ];
          }];
          metrics_path = "/metrics/slurm";
        }
      ];
      
      # Rules for alerting and aggregation
      rules = [
        ''
        groups:
        - name: node_alerts
          rules:
          - alert: HighCpuLoad
            expr: 100 - (avg by(instance) (irate(node_cpu_seconds_total{mode="idle"}[5m])) * 100) > 90
            for: 10m
            labels:
              severity: warning
            annotations:
              summary: "High CPU load on {{ $labels.instance }}"
              description: "CPU load is over 90% for more than 10 minutes on {{ $labels.instance }}"
              
          - alert: HighMemoryUsage
            expr: (node_memory_MemTotal_bytes - (node_memory_MemFree_bytes + node_memory_Buffers_bytes + node_memory_Cached_bytes)) / node_memory_MemTotal_bytes * 100 > 90
            for: 10m
            labels:
              severity: warning
            annotations:
              summary: "High memory usage on {{ $labels.instance }}"
              description: "Memory usage is over 90% for more than 10 minutes on {{ $labels.instance }}"
              
          - alert: DiskSpaceRunningOut
            expr: 100 - ((node_filesystem_avail_bytes{mountpoint="/",fstype!="rootfs"} * 100) / node_filesystem_size_bytes{mountpoint="/",fstype!="rootfs"}) < 10
            for: 10m
            labels:
              severity: warning
            annotations:
              summary: "Disk space running out on {{ $labels.instance }}"
              description: "Disk space is below 10% on {{ $labels.instance }}"
        
        - name: gpu_alerts
          rules:
          - alert: HighGpuUsage
            expr: nvidia_gpu_utilization > 95
            for: 30m
            labels:
              severity: info
            annotations:
              summary: "High GPU usage on {{ $labels.instance }}"
              description: "GPU usage is over 95% for more than 30 minutes on {{ $labels.instance }}"
              
          - alert: HighGpuTemperature
            expr: nvidia_gpu_temperature_celsius > 85
            for: 5m
            labels:
              severity: warning
            annotations:
              summary: "High GPU temperature on {{ $labels.instance }}"
              description: "GPU temperature is over 85°C for more than 5 minutes on {{ $labels.instance }}"
        ''
      ];
    };
    
    # Grafana for visualization
    services.grafana = {
      enable = true;
      port = 3000;
      addr = "0.0.0.0";
      
      # Default admin user/password
      security.adminUser = "admin";
      security.adminPasswordFile = "${pkgs.writeText "admin-password" "admin"}";
      
      # Provision data sources and dashboards
      provision = {
        enable = true;
        datasources = [
          {
            name = "Prometheus";
            type = "prometheus";
            access = "proxy";
            url = "http://localhost:9090";
            isDefault = true;
          }
        ];
        dashboards = [
          {
            name = "HPC Cluster";
            options.path = ./dashboards;
          }
        ];
      };
    };
    
    # Create dashboards directory and basic dashboards
    system.activationScripts.createGrafanaDashboards = ''
      mkdir -p ${config.services.grafana.provision.dashboards.firstOrInitial.options.path}
      
      cat > ${config.services.grafana.provision.dashboards.firstOrInitial.options.path}/node-exporter.json << 'EOF'
      {
        "annotations": { ... },
        "editable": true,
        "gnetId": 1860,
        "graphTooltip": 0,
        "id": 11,
        "iteration": 1588791896725,
        "links": [ ... ],
        "panels": [ ... ],
        "refresh": "5s",
        "schemaVersion": 22,
        "style": "dark",
        "tags": [
          "node-exporter",
          "prometheus"
        ],
        "templating": { ... },
        "time": {
          "from": "now-1h",
          "to": "now"
        },
        "timepicker": { ... },
        "timezone": "",
        "title": "Node Exporter Full",
        "uid": "rYdddlPWk",
        "version": 1
      }
      EOF
      
      cat > ${config.services.grafana.provision.dashboards.firstOrInitial.options.path}/gpu-dashboard.json << 'EOF'
      {
        "annotations": { ... },
        "editable": true,
        "gnetId": 6387,
        "graphTooltip": 0,
        "id": 12,
        "iteration": 1588791908610,
        "links": [ ... ],
        "panels": [ ... ],
        "refresh": "10s",
        "schemaVersion": 22,
        "style": "dark",
        "tags": [
          "gpu",
          "nvidia",
          "prometheus"
        ],
        "templating": { ... },
        "time": {
          "from": "now-1h",
          "to": "now"
        },
        "timepicker": { ... },
        "timezone": "",
        "title": "NVIDIA GPU",
        "uid": "AINQjHsWk",
        "version": 1
      }
      EOF
      
      cat > ${config.services.grafana.provision.dashboards.firstOrInitial.options.path}/slurm-dashboard.json << 'EOF'
      {
        "annotations": { ... },
        "editable": true,
        "gnetId": null,
        "graphTooltip": 0,
        "id": 13,
        "iteration": 1588792012345,
        "links": [ ... ],
        "panels": [ ... ],
        "refresh": "30s",
        "schemaVersion": 22,
        "style": "dark",
        "tags": [
          "slurm",
          "hpc",
          "prometheus"
        ],
        "templating": { ... },
        "time": {
          "from": "now-6h",
          "to": "now"
        },
        "timepicker": { ... },
        "timezone": "",
        "title": "SLURM Metrics",
        "uid": "SLURMdash",
        "version": 1
      }
      EOF
    '';
    
    # Alertmanager for alert handling
    services.prometheus.alertmanager = {
      enable = true;
      port = 9093;
      webExternalUrl = "http://${node.hostname}:9093";
      
      configuration = {
        global = {
          smtp_smarthost = "localhost:25";
          smtp_from = "alertmanager@${node.hostname}.${config.networking.domain}";
        };
        
        route = {
          group_by = [ "alertname", "instance" ];
          group_wait = "30s";
          group_interval = "5m";
          repeat_interval = "4h";
          receiver = "default";
          
          routes = [
            {
              match = { severity = "critical"; };
              receiver = "pager";
            }
          ];
        };
        
        receivers = [
          {
            name = "default";
            email_configs = [ 
              { 
                to = "admin@${config.networking.domain}"; 
                send_resolved = true;
              } 
            ];
          }
          {
            name = "pager";
            email_configs = [ 
              { 
                to = "oncall@${config.networking.domain}";
                send_resolved = true;
              } 
            ];
            webhook_configs = [
              {
                url = "http://localhost:8080/alert";
                send_resolved = true;
              }
            ];
          }
        ];
      };
    };
    
    # Configure Prometheus to use Alertmanager
    services.prometheus.alertmanagers = [
      {
        scheme = "http";
        path_prefix = "/";
        static_configs = [ { targets = [ "localhost:9093" ]; } ];
      }
    ];
    
    # Loki for log aggregation
    services.loki = {
      enable = true;
      configuration = {
        auth_enabled = false;
        server.http_listen_port = 3100;
        ingester = {
          lifecycler = {
            address = "127.0.0.1";
            ring = {
              kvstore.store = "inmemory";
              replication_factor = 1;
            };
            final_sleep = "0s";
          };
          chunk_idle_period = "5m";
          chunk_retain_period = "30s";
        };
        schema_config.configs = [ {
          from = "2020-05-15";
          store = "boltdb";
          object_store = "filesystem";
          schema = "v11";
          index.prefix = "index_";
          index.period = "168h";
        } ];
        storage_config = {
          boltdb.directory = "/var/lib/loki/index";
          filesystem.directory = "/var/lib/loki/chunks";
        };
        limits_config = {
          enforce_metric_name = false;
          reject_old_samples = true;
          reject_old_samples_max_age = "168h";
        };
      };
    };
    
    # Promtail to ship logs to Loki
    services.promtail = {
      enable = true;
      configuration = {
        server.http_listen_port = 9080;
        positions.filename = "/tmp/positions.yaml";
        clients = [ { url = "http://localhost:3100/loki/api/v1/push"; } ];
        scrape_configs = [ {
          job_name = "journal";
          journal = {
            max_age = "12h";
            labels = {
              job = "systemd-journal";
              host = node.hostname;
            };
          };
          relabel_configs = [
            {
              source_labels = [ "__journal__systemd_unit" ];
              target_label = "unit";
            }
          ];
        } ];
      };
    };
  };
  
  # Common monitoring configuration for all nodes
  commonMonitoring = {
    # Node exporter for system metrics
    services.prometheus.exporters.node = {
      enable = true;
      enabledCollectors = [ 
        "systemd" 
        "cpu" 
        "diskstats" 
        "filesystem" 
        "loadavg" 
        "meminfo" 
        "netdev" 
        "stat" 
        "time" 
        "vmstat" 
        "ipvs"
        "processes"
        "interrupts"
        "ksmd"
        "textfile"
      ];
      port = 9100;
    };
    
    # NVIDIA metrics exporter
    services.prometheus.exporters.nvidia = lib.mkIf hasGPU {
      enable = true;
      port = 9835;
    };
    
    # Add slurm_exporter
    systemd.services.slurm-exporter = lib.mkIf config.services.slurm.enable {
      description = "Prometheus SLURM Exporter";
      wantedBy = [ "multi-user.target" ];
      after = [ "network.target" ];
      serviceConfig = {
        ExecStart = "${pkgs.slurm-exporter}/bin/slurm-exporter --listen-address=:9102";
        User = "slurm";
        Restart = "always";
      };
    };
    
    # Log forwarding with promtail
    services.promtail = lib.mkIf (!isController) {
      enable = true;
      configuration = {
        server.http_listen_port = 9080;
        positions.filename = "/tmp/positions.yaml";
        clients = [ 
          { 
            url = "http://${(lib.findSingle (n: n.isController) (builtins.elemAt (lib.attrValues nodes) 0) (builtins.elemAt (lib.attrValues nodes) 0) (lib.attrValues nodes)).hostname}:3100/loki/api/v1/push"; 
          } 
        ];
        scrape_configs = [ {
          job_name = "journal";
          journal = {
            max_age = "12h";
            labels = {
              job = "systemd-journal";
              host = node.hostname;
            };
          };
          relabel_configs = [
            {
              source_labels = [ "__journal__systemd_unit" ];
              target_label = "unit";
            }
          ];
        } ];
      };
    };
    
    # Basic log forwarding
    services.journald.extraConfig = ''
      ForwardToSyslog=yes
    '';
    
    # Netdata for real-time monitoring (optional)
    services.netdata = lib.mkIf (config.enableNetdata or false) {
      enable = true;
      config = {
        global = {
          "update every" = "5";
          "memory mode" = "ram";
        };
        web = {
          "allow connections from" = "localhost 10.0.0.0/24";
          "allow dashboard from" = "localhost 10.0.0.0/24";
        };
      };
    };
    
    # Add monitoring utilities
    environment.systemPackages = with pkgs; [
      htop
      iotop
      sysstat
      lsof
      strace
      tcpdump
      nmap
      lm_sensors
    ] ++ lib.optionals hasGPU [
      nvtop
      cudaPackages.cuda_nvprof
    ];
    
    # Add textfile collector directory for custom metrics
    system.activationScripts.createNodeExporterTextfileDir = ''
      mkdir -p /var/lib/node_exporter/textfile_collector
      chmod 755 /var/lib/node_exporter/textfile_collector
    '';
    
    # Add SLURM metrics script
    system.activationScripts.createSlurmMetricsScript = lib.mkIf config.services.slurm.enable ''
      cat > /usr/local/bin/slurm-metrics << 'EOF'
      #!/bin/bash
      
      # Collect SLURM metrics and write to node exporter textfile collector
      OUT_FILE="/var/lib/node_exporter/textfile_collector/slurm.prom"
      
      # Get node info
      if [ -x "$(command -v sinfo)" ]; then
        NODES_ALLOC=$(sinfo -h -t alloc -o '%D')
        NODES_IDLE=$(sinfo -h -t idle -o '%D')
        NODES_DOWN=$(sinfo -h -t down -o '%D')
        
        echo "# HELP slurm_nodes_alloc Number of allocated nodes" > $OUT_FILE
        echo "# TYPE slurm_nodes_alloc gauge" >> $OUT_FILE
        echo "slurm_nodes_alloc $NODES_ALLOC" >> $OUT_FILE
        
        echo "# HELP slurm_nodes_idle Number of idle nodes" >> $OUT_FILE
        echo "# TYPE slurm_nodes_idle gauge" >> $OUT_FILE
        echo "slurm_nodes_idle $NODES_IDLE" >> $OUT_FILE
        
        echo "# HELP slurm_nodes_down Number of down nodes" >> $OUT_FILE
        echo "# TYPE slurm_nodes_down gauge" >> $OUT_FILE
        echo "slurm_nodes_down $NODES_DOWN" >> $OUT_FILE
      fi
      
      # Get job info
      if [ -x "$(command -v squeue)" ]; then
        JOBS_RUNNING=$(squeue -h -t RUNNING -o '%D' | wc -l)
        JOBS_PENDING=$(squeue -h -t PENDING -o '%D' | wc -l)
        
        echo "# HELP slurm_jobs_running Number of running jobs" >> $OUT_FILE
        echo "# TYPE slurm_jobs_running gauge" >> $OUT_FILE
        echo "slurm_jobs_running $JOBS_RUNNING" >> $OUT_FILE
        
        echo "# HELP slurm_jobs_pending Number of pending jobs" >> $OUT_FILE
        echo "# TYPE slurm_jobs_pending gauge" >> $OUT_FILE
        echo "slurm_jobs_pending $JOBS_PENDING" >> $OUT_FILE
      fi
      EOF
      
      chmod +x /usr/local/bin/slurm-metrics
    '';
    
    # Schedule SLURM metrics collection
    systemd.timers.slurm-metrics = lib.mkIf config.services.slurm.enable {
      wantedBy = [ "timers.target" ];
      timerConfig = {
        OnBootSec = "5min";
        OnUnitActiveSec = "5min";
      };
    };
    
    systemd.services.slurm-metrics = lib.mkIf config.services.slurm.enable {
      script = "/usr/local/bin/slurm-metrics";
      serviceConfig = {
        Type = "oneshot";
        User = "root";
      };
    };
  };
  
in lib.mkMerge [
  controllerMonitoring
  commonMonitoring
]
