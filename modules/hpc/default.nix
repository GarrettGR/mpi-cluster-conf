{ node }:

{ config, pkgs, lib, ... }:

let
  hasGPU = node.hasGPU or false;
  
  # Custom OpenMPI package with CUDA support
  openMPI-cuda = pkgs.openmpi.override {
    enableUCX = true;
    cudaSupport = true;
    cudatoolkit = pkgs.cudaPackages.cudatoolkit;
  };
  
  # Custom HPC environment module
  hpcEnv = pkgs.buildEnv {
    name = "hpc-env";
    paths = with pkgs; [
      openMPI-cuda
      cudaPackages.cudatoolkit
      cudaPackages.cuda_cudart
      cudaPackages.libcublas
      cudaPackages.cuda_nvml_dev
      cudaPackages.cuda_nvcc
      ucx
      numactl
      fftw
      gfortran
      gcc
      gnumake
      cmake
      binutils
      patchelf
    ];
  };
in {
  # CUDA drivers and packages
  hardware.opengl.enable = lib.mkIf hasGPU true;
  hardware.nvidia = lib.mkIf hasGPU {
    package = config.boot.kernelPackages.nvidiaPackages.stable;
    modesetting.enable = true;
    powerManagement.enable = false;
    nvidiaSettings = false;
    nvidiaPersistenced = true;
  
    # Enable CUDA development capabilities
    cudaSupport = true;
  };
  
  # Environment modules for HPC software
  environment.systemPackages = with pkgs; [
    hpcEnv
    ucx             # Unified Communication X
    numactl         # NUMA control tools
    nvidia-docker   # For containerized GPU workloads
    singularity     # Container platform for scientific computing
    lmod            # Lua-based module system
    environment-modules # Tcl-based module system
  ];
  
  # Create an optimized OpenMPI/CUDA MPI host file
  environment.etc."openmpi-hostfile".text = lib.concatMapStrings
    (n: "${n.hostname} slots=${if (n.hasGPU or false) then toString (n.cpus or 64) else "1"}\n")
    (lib.attrValues config.cluster.nodes or []);
    
  # Configure UCX for optimal CUDA performance
  environment.variables = {
    UCX_TLS = "rc,sm,cuda_copy,cuda_ipc,gdr_copy";
    UCX_RNDV_SCHEME = "get_zcopy";
    UCX_MEMTYPE_CACHE = "n";
    UCX_IB_GPU_DIRECT_RDMA = lib.mkIf hasGPU "yes";
    
    # OpenMPI CUDA-aware settings
    OMPI_MCA_pml = "ucx";
    OMPI_MCA_btl = "^openib,vader,tcp,uct";
    OMPI_MCA_osc = "ucx";
    
    # Add the HPC environment to the path
    PATH = lib.mkBefore [ "${hpcEnv}/bin" ];
    LD_LIBRARY_PATH = lib.mkBefore [ "${hpcEnv}/lib" ];
  };
  
  # Create CPU and GPU affinity scripts
  system.activationScripts.gpuHelperScripts = lib.mkIf hasGPU ''
    cat > /usr/local/bin/gpu-affinity.sh << 'EOF'
    #!/bin/bash
    # Helper script to set GPU-CPU affinity for optimal performance
    export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
    export UCX_NET_DEVICES=mlx5_0:1
    export UCX_TLS=rc,sm,cuda_copy,cuda_ipc,gdr_copy
    exec "$@"
    EOF
    
    chmod +x /usr/local/bin/gpu-affinity.sh
  '';
  
  # Scientific Python packages for data analysis
  environment.systemPackages = with pkgs; [
    (python3.withPackages (ps: with ps; [
      numpy
      scipy
      matplotlib
      pandas
      scikit-learn
      ipython
      jupyter
      mpi4py
    ]))
  ];
  
  # CUDA optimized libraries
  environment.systemPackages = lib.mkIf hasGPU (with pkgs; [
    cudaPackages.cudnn
    cudaPackages.nccl
    cudaPackages.cutensor
    magma # Matrix Algebra for GPU and Multicore Architectures
  ]);
  
  # Add compiler toolchains
  environment.systemPackages = with pkgs; [
    gcc
    clang
    gfortran
    gnumake
    cmake
    ninja
    automake
    autoconf
    libtool
    pkg-config
  ];
  
  # HPC benchmark tools
  environment.systemPackages = with pkgs; [
    hpl # High-Performance Linpack
    stream # Memory bandwidth benchmark
    ior # IO benchmark
    fio # Flexible IO tester
    intel-mpi-benchmarks # MPI benchmarking
    stressapptest # Memory interface test
    sysbench # System performance benchmark
  ] ++ lib.optionals hasGPU [
    cudaPackages.cuda_samples # CUDA samples and benchmarks
  ];
  
  # Math libraries
  environment.systemPackages = with pkgs; [
    openblas
    mkl # Intel Math Kernel Library
    fftw
    gsl # GNU Scientific Library
    lapack
    scalapack
    arpack
  ];
  
  # Additional container tools
  virtualisation = {
    docker = {
      enable = true;
      enableNvidia = hasGPU;
    };
    
    singularity = {
      enable = true;
      enableNvidia = hasGPU;
    };
  };
  
  # CPU frequency scaling for performance
  powerManagement.cpuFreqGovernor = "performance";
  
  # Increase limits for HPC workloads
  security.pam.loginLimits = [
    { domain = "*"; type = "soft"; item = "nofile"; value = "1048576"; }
    { domain = "*"; type = "hard"; item = "nofile"; value = "1048576"; }
    { domain = "*"; type = "soft"; item = "nproc"; value = "unlimited"; }
    { domain = "*"; type = "hard"; item = "nproc"; value = "unlimited"; }
    { domain = "*"; type = "soft"; item = "memlock"; value = "unlimited"; }
    { domain = "*"; type = "hard"; item = "memlock"; value = "unlimited"; }
  ];
  
  # Create HPC environment modules
  environment.modules.packages = with pkgs; [
    openmpi
    openMPI-cuda
    fftw
    openblas
    python3
    cudaPackages.cudatoolkit
  ];
  
  # NUMA optimizations
  hardware.cpu.amd.updateMicrocode = lib.mkDefault config.hardware.enableRedistributableFirmware;
  hardware.cpu.intel.updateMicrocode = lib.mkDefault config.hardware.enableRedistributableFirmware;
  
  # CPU performance settings
  boot.kernelModules = [ "msr" ];
  boot.kernelParams = [
    "intel_pstate=disable" # Use acpi-cpufreq instead
    "processor.max_cstate=1" # Disable deep sleep states
    "idle=poll" # Disable CPU idle
    "intel_idle.max_cstate=0" # Disable Intel idle driver
    "nosmt" # Disable Simultaneous Multi-Threading
    "nohz_full=1-${toString ((node.cpus or 64) - 1)}" # Disable timer interrupts on CPU cores
    "isolcpus=1-${toString ((node.cpus or 64) - 1)}" # Isolate CPUs
  ];
  
  # Module for Chapel language (optional, for programming in Chapel)
  # environment.systemPackages = lib.mkIf (config.enableChapel or false) (with pkgs; [
  #   chapel
  # ]);
  
  # Julia language for scientific computing
  environment.systemPackages = lib.mkIf (config.enableJulia or false) (with pkgs; [
    julia-bin
  ]);
  
  # R for statistical computing
  environment.systemPackages = lib.mkIf (config.enableR or false) (with pkgs; [
    R
    rPackages.tidyverse
    rPackages.data_table
  ]);
}
