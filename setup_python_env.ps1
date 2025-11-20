# Python Environment Wrapper Script
# Use this to run Python commands with the correct Python installation

$PYTHON311 = "C:\Users\Viraj Jain\AppData\Local\Programs\Python\Python311\python.exe"

# Show instructions
Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host "Python 3.11 Environment Wrapper" -ForegroundColor Cyan
Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Using: Python 3.11 (with PyTorch + CUDA)" -ForegroundColor Green
Write-Host "Path: $PYTHON311" -ForegroundColor Gray
Write-Host ""

# Check if Python exists
if (-not (Test-Path $PYTHON311)) {
    Write-Host "ERROR: Python 3.11 not found!" -ForegroundColor Red
    exit 1
}

# Show available commands
Write-Host "Available Commands:" -ForegroundColor Yellow
Write-Host "  python <script.py>          - Run Python script"
Write-Host "  pytest                      - Run tests"
Write-Host "  pip install <package>       - Install package"
Write-Host ""

# Set alias for current session
Write-Host "Setting up aliases for this session..." -ForegroundColor Green
Set-Alias -Name python -Value $PYTHON311 -Scope Global -Force
Set-Alias -Name python3 -Value $PYTHON311 -Scope Global -Force

Write-Host "✓ Aliases configured!" -ForegroundColor Green
Write-Host ""
Write-Host "You can now use 'python' and it will use Python 3.11" -ForegroundColor Cyan
Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host ""

# Test PyTorch
Write-Host "Testing PyTorch installation..." -ForegroundColor Yellow
& $PYTHON311 -c "import torch; print(f'PyTorch {torch.__version__} (CUDA: {torch.cuda.is_available()})')"
Write-Host ""
