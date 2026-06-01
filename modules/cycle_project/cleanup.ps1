# SUBSTRATE — limpieza de estructura obsoleta
# Ejecutar desde: C:\Users\Drizzy\Desktop\cycle_project
# powershell -ExecutionPolicy Bypass -File cleanup.ps1

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

Write-Host "Limpiando estructura obsoleta de SUBSTRATE..." -ForegroundColor Cyan

# Capas obsoletas
Remove-Item -Recurse -Force "substrate"        -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force "notebooks"        -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force "scripts"          -ErrorAction SilentlyContinue
Remove-Item -Force         "run_real_data.py"  -ErrorAction SilentlyContinue
Remove-Item -Force         "download_real_data.py" -ErrorAction SilentlyContinue
Remove-Item -Force         "pyproject.toml"    -ErrorAction SilentlyContinue

# Scripts sueltos en src
Remove-Item -Force "src\cycle_detect\generate_synthetic.py" -ErrorAction SilentlyContinue

# src/field_coherence (stub vacío, el bueno es field_coherence_monitor)
Remove-Item -Recurse -Force "src\field_coherence" -ErrorAction SilentlyContinue

# Pycache
Get-ChildItem -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
Get-ChildItem -Recurse -Filter "*.pyc" | Remove-Item -Force -ErrorAction SilentlyContinue

Write-Host "Hecho." -ForegroundColor Green
Write-Host ""
Write-Host "Estructura resultante:" -ForegroundColor Yellow
Get-ChildItem -Depth 2 | Where-Object { $_.Name -notin @("data","target",".git") }
