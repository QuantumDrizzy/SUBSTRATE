import os
import logging
from pathlib import Path

logger = logging.getLogger("substrate.dll_healing")

_healed = False  # guard: os.add_dll_directory() accumulates on every call

def heal():
    """
    Ensures that CUDA and Nvidia DLLs are found by the Python interpreter on Windows.
    This is critical for libraries like CuPy and cuQuantum when running inside an 
    embedded environment or specialized conda/pip setups.
    """
    global _healed
    if _healed:
        return
    if os.name != 'nt':
        _healed = True
        return

    logger.debug("Initializing Windows DLL healing...")
    
    # 1. Check CUDA_PATH environment variables
    cuda_keys = ['CUDA_PATH', 'CUDA_PATH_V13_0', 'CUDA_PATH_V12_5', 'CUDA_PATH_V12_4']
    for key in cuda_keys:
        path = os.environ.get(key)
        if path:
            bin_path = Path(path) / "bin"
            if bin_path.exists():
                logger.debug(f"[DLL] Adding {key}: {bin_path}")
                os.add_dll_directory(str(bin_path))

    # 2. Check common default locations
    defaults = [
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin",
    ]
    for d in defaults:
        if Path(d).exists():
            logger.debug(f"[DLL] Adding default: {d}")
            os.add_dll_directory(d)

    # 3. Check site-packages (nvidia-* wheels)
    try:
        import site
        search_paths = site.getsitepackages()
        if site.getusersitepackages():
            search_paths.append(site.getusersitepackages())
            
        for sp in search_paths:
            nvidia_path = Path(sp) / "nvidia"
            if nvidia_path.exists():
                subpackages = [
                    "cublas", "cusolver", "cufft", "curand", 
                    "cusparse", "cuda_runtime", "cuda_nvrtc", "cudnn",
                    "nvjitlink", "nvfatbin"
                ]
                for sub in subpackages:
                    bin_path = nvidia_path / sub / "bin"
                    if bin_path.exists():
                        logger.debug(f"[DLL] Adding site-package: {bin_path}")
                        os.add_dll_directory(str(bin_path))
    except Exception as e:
        logger.warning(f"DLL healing site-package scan failed: {e}")

    _healed = True

# Run healing on import
heal()
