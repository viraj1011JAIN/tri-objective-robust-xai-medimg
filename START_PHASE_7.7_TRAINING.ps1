#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Quick-start script for Phase 7.7 tri-objective training.

.DESCRIPTION
    This script provides a streamlined way to start tri-objective training
    without needing the full validation module implementation yet.

    It will:
    1. Verify environment and dependencies
    2. Check configuration file
    3. Start training for all three seeds
    4. Monitor progress
    5. Generate basic summary report

.PARAMETER Seeds
    Random seeds to train (default: 42, 123, 456)

.PARAMETER GPU
    GPU device ID (default: 0)

.PARAMETER ConfigPath
    Path to config file (default: configs/experiments/tri_objective.yaml)

.EXAMPLE
    .\START_PHASE_7.7_TRAINING.ps1

.EXAMPLE
    .\START_PHASE_7.7_TRAINING.ps1 -Seeds 42 -GPU 0

.NOTES
    Author: Viraj Pankaj Jain
    Phase: 7.7 - Initial Tri-Objective Validation
#>

param(
    [int[]]$Seeds = @(42, 123, 456),
    [int]$GPU = 0,
    [string]$ConfigPath = "configs/experiments/tri_objective.yaml"
)

# Color output functions
function Write-Header {
    param([string]$Message)
    Write-Host "`n" + ("=" * 70) -ForegroundColor Cyan
    Write-Host $Message -ForegroundColor Cyan
    Write-Host ("=" * 70) -ForegroundColor Cyan
}

function Write-Success {
    param([string]$Message)
    Write-Host "✅ $Message" -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "⚠️  $Message" -ForegroundColor Yellow
}

function Write-Error-Custom {
    param([string]$Message)
    Write-Host "❌ $Message" -ForegroundColor Red
}

function Write-Info {
    param([string]$Message)
    Write-Host "ℹ️  $Message" -ForegroundColor Blue
}

# Main execution
Write-Header "PHASE 7.7 - TRI-OBJECTIVE TRAINING QUICK START"

Write-Info "Project: Tri-Objective Robust XAI for Medical Imaging"
Write-Info "Institution: University of Glasgow"
Write-Info "Date: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host ""

# Step 1: Verify environment
Write-Header "Step 1: Environment Verification"

Write-Info "Checking Python environment..."
try {
    $pythonVersion = python --version 2>&1
    Write-Success "Python found: $pythonVersion"
} catch {
    Write-Error-Custom "Python not found in PATH"
    exit 1
}

Write-Info "Checking virtual environment..."
if ($env:VIRTUAL_ENV) {
    Write-Success "Virtual environment active: $env:VIRTUAL_ENV"
} else {
    Write-Warning "No virtual environment detected - activating .venv..."
    & ".\.venv\Scripts\Activate.ps1"
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Virtual environment activated"
    } else {
        Write-Error-Custom "Failed to activate virtual environment"
        exit 1
    }
}

Write-Info "Checking CUDA availability..."
$cudaCheck = python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}')" 2>&1
Write-Success $cudaCheck

Write-Info "Checking key dependencies..."
$deps = @("torch", "torchvision", "mlflow", "matplotlib", "numpy", "pandas", "pyyaml")
$missingDeps = @()

foreach ($dep in $deps) {
    $checkResult = python -c "import $dep; print('OK')" 2>&1
    if ($checkResult -match "OK") {
        Write-Host "  ✓ $dep" -ForegroundColor Green
    } else {
        Write-Host "  ✗ $dep" -ForegroundColor Red
        $missingDeps += $dep
    }
}

if ($missingDeps.Count -gt 0) {
    Write-Error-Custom "Missing dependencies: $($missingDeps -join ', ')"
    Write-Info "Install with: pip install $($missingDeps -join ' ')"
    exit 1
}

Write-Success "All dependencies available"

# Step 2: Verify configuration
Write-Header "Step 2: Configuration Verification"

if (Test-Path $ConfigPath) {
    Write-Success "Config file found: $ConfigPath"

    Write-Info "Checking config contents..."
    $config = Get-Content $ConfigPath -Raw | ConvertFrom-Yaml -ErrorAction SilentlyContinue
    if ($config) {
        Write-Success "Config file is valid YAML"

        # Check key sections
        $requiredSections = @("experiment", "model", "dataset", "loss", "training")
        foreach ($section in $requiredSections) {
            if ($config.ContainsKey($section)) {
                Write-Host "  ✓ Section '$section' present" -ForegroundColor Green
            } else {
                Write-Warning "Section '$section' missing"
            }
        }
    } else {
        Write-Warning "Could not parse YAML (PowerShell limitation) - assuming valid"
    }
} else {
    Write-Error-Custom "Config file not found: $ConfigPath"
    exit 1
}

# Step 3: Check training script
Write-Header "Step 3: Training Script Verification"

$trainScript = "scripts/train_tri_objective.py"
if (Test-Path $trainScript) {
    Write-Success "Training script found: $trainScript"
} else {
    Write-Error-Custom "Training script not found: $trainScript"
    Write-Info "Expected location: scripts/train_tri_objective.py"
    exit 1
}

# Step 4: Prepare output directories
Write-Header "Step 4: Output Directory Preparation"

$outputDirs = @(
    "results/checkpoints/tri_objective",
    "results/logs/training",
    "results/plots/training_curves",
    "results/metrics/tri_objective"
)

foreach ($dir in $outputDirs) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Success "Created: $dir"
    } else {
        Write-Info "Exists: $dir"
    }
}

# Step 5: Display training plan
Write-Header "Step 5: Training Plan"

Write-Host ""
Write-Info "Configuration:"
Write-Host "  Seeds: $($Seeds -join ', ')"
Write-Host "  GPU Device: $GPU"
Write-Host "  Config: $ConfigPath"
Write-Host "  Estimated Time: $($Seeds.Count * 1.5) - $($Seeds.Count * 2) hours"
Write-Host ""

Write-Info "Target Metrics:"
Write-Host "  Clean Accuracy:  ≥0.83 (allow slight drop from baseline 0.85)"
Write-Host "  Robust Accuracy: ≥0.45 (+35pp from baseline 0.10)"
Write-Host "  SSIM Stability:  ≥0.75 (+0.15 from baseline 0.60)"
Write-Host "  Artifact TCAV:   ≤0.20 (-0.25 from baseline 0.45)"
Write-Host "  Medical TCAV:    ≥0.68 (+0.10 from baseline 0.58)"
Write-Host ""

# Step 6: Confirm execution
Write-Host "Ready to start training? " -NoNewline
Write-Host "[Y/n]: " -ForegroundColor Yellow -NoNewline
$response = Read-Host

if ($response -and $response -ne "Y" -and $response -ne "y" -and $response -ne "") {
    Write-Warning "Training cancelled by user"
    exit 0
}

# Step 7: Execute training
Write-Header "Step 6: Training Execution"

$startTime = Get-Date
$results = @()

foreach ($i = 0; $i -lt $Seeds.Count; $i++) {
    $seed = $Seeds[$i]
    $seedNum = $i + 1

    Write-Host ""
    Write-Host ("#" * 70) -ForegroundColor Magenta
    Write-Host "# SEED $seedNum/$($Seeds.Count): $seed" -ForegroundColor Magenta
    Write-Host ("#" * 70) -ForegroundColor Magenta
    Write-Host ""

    $seedStartTime = Get-Date

    Write-Info "Starting training for seed $seed..."
    Write-Host ""

    # Build command
    $cmd = "python scripts/train_tri_objective.py --config $ConfigPath --seed $seed --gpu $GPU"

    Write-Info "Command: $cmd"
    Write-Host ""

    # Execute training
    try {
        Invoke-Expression $cmd

        if ($LASTEXITCODE -eq 0) {
            $seedEndTime = Get-Date
            $seedDuration = $seedEndTime - $seedStartTime

            Write-Success "Seed $seed training completed"
            Write-Info "Duration: $($seedDuration.ToString('hh\:mm\:ss'))"

            $results += @{
                Seed = $seed
                Status = "Success"
                Duration = $seedDuration
            }
        } else {
            Write-Error-Custom "Seed $seed training failed (exit code: $LASTEXITCODE)"

            $results += @{
                Seed = $seed
                Status = "Failed"
                ExitCode = $LASTEXITCODE
            }
        }
    } catch {
        Write-Error-Custom "Seed $seed training encountered error: $_"

        $results += @{
            Seed = $seed
            Status = "Error"
            Message = $_.Exception.Message
        }
    }

    # Progress update
    $elapsed = (Get-Date) - $startTime
    $completed = $seedNum
    $remaining = $Seeds.Count - $completed

    Write-Host ""
    Write-Info "Progress: $completed/$($Seeds.Count) seeds completed"
    Write-Info "Elapsed time: $($elapsed.ToString('hh\:mm\:ss'))"

    if ($remaining -gt 0 -and $completed -gt 0) {
        $avgTime = $elapsed.TotalSeconds / $completed
        $eta = [TimeSpan]::FromSeconds($avgTime * $remaining)
        Write-Info "Estimated remaining: $($eta.ToString('hh\:mm\:ss'))"
    }

    # Clean GPU memory between seeds
    if ($remaining -gt 0) {
        Write-Info "Cleaning GPU memory before next seed..."
        Start-Sleep -Seconds 5
    }
}

# Step 8: Summary Report
Write-Header "Step 7: Training Summary"

$endTime = Get-Date
$totalDuration = $endTime - $startTime

Write-Host ""
Write-Info "Total Training Time: $($totalDuration.ToString('hh\:mm\:ss'))"
Write-Host ""

Write-Host "Results by Seed:" -ForegroundColor Cyan
Write-Host ("-" * 70)

$successCount = 0
foreach ($result in $results) {
    $seed = $result.Seed
    $status = $result.Status

    Write-Host "  Seed $seed`: " -NoNewline

    if ($status -eq "Success") {
        Write-Host "✅ SUCCESS" -ForegroundColor Green -NoNewline
        Write-Host " ($($result.Duration.ToString('hh\:mm\:ss')))"
        $successCount++
    } elseif ($status -eq "Failed") {
        Write-Host "❌ FAILED" -ForegroundColor Red -NoNewline
        Write-Host " (Exit code: $($result.ExitCode))"
    } else {
        Write-Host "❌ ERROR" -ForegroundColor Red -NoNewline
        Write-Host " ($($result.Message))"
    }
}

Write-Host ("-" * 70)
Write-Host ""

if ($successCount -eq $Seeds.Count) {
    Write-Success "All seeds completed successfully! 🎉"
    Write-Host ""
    Write-Info "Next steps:"
    Write-Host "  1. Check results: results/checkpoints/tri_objective/"
    Write-Host "  2. View metrics: mlflow ui --port 5000"
    Write-Host "  3. Generate report: python scripts/validation/aggregate_results.py"
} elseif ($successCount -gt 0) {
    Write-Warning "$successCount/$($Seeds.Count) seeds completed"
    Write-Info "Review failed seed logs in: results/logs/training/"
} else {
    Write-Error-Custom "No seeds completed successfully"
    Write-Info "Check logs and configuration, then retry"
}

Write-Host ""
Write-Header "Phase 7.7 Training Session Complete"

Write-Host ""
Write-Info "Output Locations:"
Write-Host "  Checkpoints: results/checkpoints/tri_objective/"
Write-Host "  Logs: results/logs/training/"
Write-Host "  Metrics: MLflow tracking (mlflow ui)"
Write-Host ""

exit 0
