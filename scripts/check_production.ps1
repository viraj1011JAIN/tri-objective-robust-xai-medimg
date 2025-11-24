# Production Readiness Quick Check
# Simplified verification script with ASCII-only characters

$ErrorActionPreference = "Stop"

Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host "    Production Readiness Verification v1.0" -ForegroundColor Cyan
Write-Host "    Tri-Objective Robust XAI for Medical Imaging" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "`n"

$passed = 0
$failed = 0
$total = 0

function Test-Item {
    param([string]$Name, [scriptblock]$Check)
    $script:total++
    Write-Host "  $Name..." -NoNewline
    try {
        $result = & $Check
        if ($result) {
            Write-Host " PASS" -ForegroundColor Green
            $script:passed++
        } else {
            Write-Host " FAIL" -ForegroundColor Red
            $script:failed++
        }
    } catch {
        Write-Host " FAIL ($($_.Exception.Message))" -ForegroundColor Red
        $script:failed++
    }
}

# 1. Environment
Write-Host "`n--- Environment Setup ---" -ForegroundColor Yellow
Test-Item "Python 3.11+" { (python --version 2>&1) -match "Python 3\.1[1-9]" }
Test-Item "PyTorch installed" { (python -c "import torch; print('OK')" 2>&1) -match "OK" }
Test-Item "CUDA available" { (python -c "import torch; print(torch.cuda.is_available())" 2>&1) -match "True" }
Test-Item "requirements.txt" { Test-Path "requirements.txt" }
Test-Item "environment.yml" { Test-Path "environment.yml" }
Test-Item "pyproject.toml" { Test-Path "pyproject.toml" }

# 2. MLOps
Write-Host "`n--- MLOps Infrastructure ---" -ForegroundColor Yellow
Test-Item "DVC initialized" { Test-Path ".dvc" }
Test-Item "DVC remote configured" { Test-Path ".dvc_storage" }
Test-Item "MLflow directory" { Test-Path "mlruns" }

# 3. Code Quality
Write-Host "`n--- Code Quality and CI/CD ---" -ForegroundColor Yellow
Test-Item "Pre-commit config" { Test-Path ".pre-commit-config.yaml" }
Test-Item "GitHub Actions tests" { Test-Path ".github/workflows/tests.yml" }
Test-Item "GitHub Actions lint" { Test-Path ".github/workflows/lint.yml" }
Test-Item "GitHub Actions docs" { Test-Path ".github/workflows/docs.yml" }

# 4. Testing
Write-Host "`n--- Testing Infrastructure ---" -ForegroundColor Yellow
Test-Item "pytest.ini" { Test-Path "pytest.ini" }
Test-Item "tests directory" { Test-Path "tests" }
Test-Item "Test files exist" { (Get-ChildItem -Path "tests" -Filter "test_*.py" -Recurse).Count -gt 0 }

# 5. Documentation
Write-Host "`n--- Documentation ---" -ForegroundColor Yellow
Test-Item "README.md" { Test-Path "README.md" }
Test-Item "CONTRIBUTING.md" { Test-Path "CONTRIBUTING.md" }
Test-Item "LICENSE" { Test-Path "LICENSE" }
Test-Item "CITATION.cff" { Test-Path "CITATION.cff" }
Test-Item "docs directory" { Test-Path "docs" }

# 6. Project Structure
Write-Host "`n--- Project Structure ---" -ForegroundColor Yellow
$requiredDirs = @("src", "configs", "data", "logs", "results", "scripts", "tests", "docs")
foreach ($dir in $requiredDirs) {
    Test-Item "Directory: $dir" { Test-Path $dir }
}

# 7. Configuration
Write-Host "`n--- Configuration Files ---" -ForegroundColor Yellow
Test-Item "Base config" { Test-Path "configs/base.yaml" }
Test-Item "Dataset configs" { (Get-ChildItem -Path "configs/datasets" -Filter "*.yaml").Count -gt 0 }
Test-Item "Model configs" { (Get-ChildItem -Path "configs/models" -Filter "*.yaml").Count -gt 0 }

# 8. Reproducibility
Write-Host "`n--- Reproducibility ---" -ForegroundColor Yellow
Test-Item "Reproducibility module" { Test-Path "src/utils/reproducibility.py" }
Test-Item "Config management" { Test-Path "src/utils/config.py" }

# Summary
Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host "    VERIFICATION SUMMARY" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Total Checks: $total" -ForegroundColor White
Write-Host "Passed: $passed" -ForegroundColor Green
Write-Host "Failed: $failed" -ForegroundColor Red
$percentage = [math]::Round(($passed / $total) * 100, 1)
Write-Host "Success Rate: $percentage%" -ForegroundColor $(if ($percentage -eq 100) { "Green" } elseif ($percentage -ge 90) { "Yellow" } else { "Red" })

Write-Host "`n"
if ($percentage -eq 100) {
    Write-Host "SUCCESS: PRODUCTION READY! All checks passed." -ForegroundColor Green
    exit 0
} elseif ($percentage -ge 90) {
    Write-Host "WARNING: Mostly ready. Some items need attention." -ForegroundColor Yellow
    exit 0
} else {
    Write-Host "ERROR: NOT READY. Critical items missing." -ForegroundColor Red
    exit 1
}
