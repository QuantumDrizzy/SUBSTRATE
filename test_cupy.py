import sys
from pathlib import Path

# Try to use our new DLL healing
sys.path.append(str(Path.cwd() / "engine"))
try:
    import dll_healing
    print("DLL Healing initialized.")
except ImportError:
    print("Could not import dll_healing.")

try:
    import cupy
    print(f"Cupy imported! Device: {cupy.cuda.Device().compute_capability}")
except Exception as e:
    print(f"Cupy FAILED: {e}")
