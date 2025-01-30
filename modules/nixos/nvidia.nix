{config, pkgs, ...}:
  hardware = {
    graphics.enable = true;
    nvidia = {
      package = config.boot.kernelPackages.nvidiaPackages.production;
      modesetting.enable = true;
      nvidiaSettings = true;
    };
  };

  services.xserver.videoDrivers = [ "nvidia" ];

  environment.systemPackages = with pkgs; [
    cudaPackages.nccl
    cudaPackages.cudatoolkit
    cudaPackages.cuda_opencl
    cudaPackages.cuda_nvcc
    cudaPackages.cuda_gdb
    cudaPackages.cuda_cudart
    cudaPackages.backendStdenv
    nvtopPackages.nvidia
  ];
}
