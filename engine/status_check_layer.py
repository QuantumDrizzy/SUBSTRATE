import os
import sys
import logging
from pathlib import Path

# Add engine directory to path to import dll_healing
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def run(params: dict = None) -> dict:
    """
    Performs a system-wide check of dependencies and GPU availability.
    Used by 'substrate check'.
    """
    try:
        from dll_healing import heal
        heal()
        
        results = {
            "os": os.name,
            "python_version": sys.version,
            "cuda_available": False,
            "gpu_count": 0,
            "cupy_ok": False,
            "quimb_ok": False,
            "torch_ok": False,
        }

        # Check CuPy
        try:
            import cupy
            results["cupy_ok"] = True
            results["cuda_available"] = cupy.is_available()
            if results["cuda_available"]:
                results["gpu_count"] = cupy.cuda.runtime.getDeviceCount()
                try:
                    results["gpu_name"] = cupy.cuda.Device(0).name
                except:
                    results["gpu_name"] = "NVIDIA GPU"
        except ImportError:
            pass

        # Check Quimb
        try:
            import quimb
            results["quimb_ok"] = True
            results["quimb_version"] = quimb.__version__
        except ImportError:
            pass

        # Check PyTorch (optional but useful for some models)
        try:
            import torch
            results["torch_ok"] = True
            results["torch_cuda"] = torch.cuda.is_available()
        except ImportError:
            pass

        print("\n--- Diagnostic Report ---")
        print(f"CUDA Available: {results['cuda_available']}")
        if results['cuda_available']:
            print(f"GPU Found:      {results.get('gpu_name', 'Unknown')}")
        print(f"CuPy Status:    {'READY' if results['cupy_ok'] else 'NOT FOUND'}")
        print(f"Quimb Status:   {'READY' if results['quimb_ok'] else 'NOT FOUND'}")
        print("------------------------\n")

        return {
            "score": 1.0 if results["cuda_available"] else 0.5,
            "data": results
        }
    except Exception as e:
        return {
            "score": 0.0,
            "error": str(e)
        }
