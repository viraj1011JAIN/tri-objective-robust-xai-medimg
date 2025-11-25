# =============================================================================
# Phase 5.4 Complete HPO Pipeline Automation Script
# =============================================================================
#
# This PowerShell script automates the complete TRADES hyperparameter
# optimization workflow including:
#   1. HPO study execution with Optuna
#   2. Results analysis and visualization
#   3. Model retraining with optimal hyperparameters
#
# Author: Viraj Pankaj Jain
# Institution: University of Glasgow, School of Computing Science
# Date: November 24, 2025
# Version: 5.4.0
# =============================================================================

[CmdletBinding()]
param(
    # Execution mode flags
    [switch]$QuickTest,
    [switch]$SkipHPO,
    [switch]$SkipRetrain,
    [switch]$SkipAnalysis,

    # HPO configuration
    [string]$StudyName = "trades_hpo_phase54",
    [int]$NTrials = 50,
    [int]$NEpochs = 10,
    [string]$Storage = "sqlite:///hpo_study.db",

    # Retrain configuration
    [int]$RetrainEpochs = 200,
    [switch]$UseScheduler,
    [switch]$UseMLflow,

    # Data configuration
    [string]$Dataset = "cifar10",
    [string]$DataDir = "data",
    [int]$BatchSize = 128,
    [int]$NumWorkers = 4,

    # Model configuration
    [string]$Model = "resnet18",

    # Device configuration
    [string]$Device = "auto",
    [int]$GpuId = 0,

    # Output configuration
    [string]$OutputDir = "results/phase_5_4",
    [string]$CheckpointDir = "checkpoints/hpo",
    [string]$FinalCheckpointDir = "checkpoints/final_model"
)

# =============================================================================
# Script Configuration
# =============================================================================

$ErrorActionPreference = "Stop"
$ProgressPreference = "Continue"

# Colors for output
$ColorInfo = "Cyan"
$ColorSuccess = "Green"
$ColorWarning = "Yellow"
$ColorError = "Red"
$ColorHeader = "Magenta"

# Timing
$ScriptStartTime = Get-Date

# =============================================================================
# Helper Functions
# =============================================================================

function Write-Header {
    param([string]$Message)
    Write-Host ""
    Write-Host ("=" * 80) -ForegroundColor $ColorHeader
    Write-Host $Message -ForegroundColor $ColorHeader
    Write-Host ("=" * 80) -ForegroundColor $ColorHeader
    Write-Host ""
}

function Write-Info {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor $ColorInfo
}

function Write-Success {
    param([string]$Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor $ColorSuccess
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor $ColorWarning
}

function Write-ErrorMsg {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor $ColorError
}

function Test-PythonEnvironment {
    Write-Info "Checking Python environment..."

    try {
        $pythonVersion = python --version 2>&1
        Write-Success "Python found: $pythonVersion"

        # Check required packages
        $requiredPackages = @("torch", "optuna", "numpy", "pandas", "matplotlib", "plotly")
        foreach ($package in $requiredPackages) {
            python -c "import $package" 2>&1 | Out-Null
            if ($LASTEXITCODE -eq 0) {
                Write-Success "Package '$package' is available"
            } else {
                Write-Warning "Package '$package' not found. Install with: pip install $package"
            }
        }

        return $true
    } catch {
        Write-ErrorMsg "Python environment check failed: $_"
        return $false
    }
}

function Invoke-HPOStudy {
    Write-Header "STEP 1: Running HPO Study"

    $hpoArgs = @(
        "scripts/run_hpo_study.py",
        "--study-name", $StudyName,
        "--n-trials", $NTrials,
        "--n-epochs", $NEpochs,
        "--storage", $Storage,
        "--dataset", $Dataset,
        "--data-dir", $DataDir,
        "--batch-size", $BatchSize,
        "--num-workers", $NumWorkers,
        "--model", $Model,
        "--device", $Device,
        "--gpu-id", $GpuId,
        "--output-dir", $OutputDir,
        "--checkpoint-dir", $CheckpointDir
    )

    if ($QuickTest) {
        $hpoArgs += "--quick-test"
    }

    if ($SkipAnalysis) {
        $hpoArgs += "--skip-analysis"
    }

    Write-Info "Executing HPO study with $NTrials trials..."
    Write-Info "Command: python $($hpoArgs -join ' ')"

    $hpoStartTime = Get-Date

    try {
        & python @hpoArgs

        if ($LASTEXITCODE -ne 0) {
            throw "HPO study failed with exit code $LASTEXITCODE"
        }

        $hpoElapsed = (Get-Date) - $hpoStartTime
        Write-Success "HPO study completed in $($hpoElapsed.ToString('hh\:mm\:ss'))"

    } catch {
        Write-ErrorMsg "HPO study execution failed: $_"
        throw
    }
}

function Invoke-ModelRetrain {
    Write-Header "STEP 2: Retraining with Optimal Hyperparameters"

    $retrainArgs = @(
        "scripts/retrain_optimal.py",
        "--study-name", $StudyName,
        "--storage", $Storage,
        "--n-epochs", $RetrainEpochs,
        "--dataset", $Dataset,
        "--data-dir", $DataDir,
        "--batch-size", $BatchSize,
        "--num-workers", $NumWorkers,
        "--model", $Model,
        "--device", $Device,
        "--gpu-id", $GpuId,
        "--output-dir", "$OutputDir/final_model",
        "--checkpoint-dir", $FinalCheckpointDir
    )

    if ($UseScheduler) {
        $retrainArgs += "--use-scheduler"
    }

    if ($UseMLflow) {
        $retrainArgs += "--use-mlflow"
    }

    Write-Info "Retraining model for $RetrainEpochs epochs..."
    Write-Info "Command: python $($retrainArgs -join ' ')"

    $retrainStartTime = Get-Date

    try {
        & python @retrainArgs

        if ($LASTEXITCODE -ne 0) {
            throw "Model retraining failed with exit code $LASTEXITCODE"
        }

        $retrainElapsed = (Get-Date) - $retrainStartTime
        Write-Success "Model retraining completed in $($retrainElapsed.ToString('hh\:mm\:ss'))"

    } catch {
        Write-ErrorMsg "Model retraining failed: $_"
        throw
    }
}

function Show-Summary {
    Write-Header "PHASE 5.4 EXECUTION SUMMARY"

    $totalElapsed = (Get-Date) - $ScriptStartTime

    Write-Info "Total execution time: $($totalElapsed.ToString('hh\:mm\:ss'))"
    Write-Info ""

    # Show study summary if exists
    $summaryPath = Join-Path $OutputDir "analysis/hpo_summary.json"
    if (Test-Path $summaryPath) {
        Write-Info "HPO Study Summary:"
        $summary = Get-Content $summaryPath | ConvertFrom-Json
        Write-Info "  Study name: $($summary.study_name)"
        Write-Info "  Total trials: $($summary.n_trials)"
        Write-Info "  Completed: $($summary.n_complete)"
        Write-Info "  Pruned: $($summary.n_pruned)"
        Write-Info "  Best value: $($summary.best_value)"
        Write-Info "  Best parameters:"
        foreach ($param in $summary.best_params.PSObject.Properties) {
            Write-Info "    $($param.Name): $($param.Value)"
        }
        Write-Info ""
    }

    # Show retrain summary if exists
    $retrainSummaryPath = Join-Path $OutputDir "final_model/training_summary.json"
    if (Test-Path $retrainSummaryPath) {
        Write-Info "Final Model Training Summary:"
        $retrainSummary = Get-Content $retrainSummaryPath | ConvertFrom-Json
        Write-Info "  Best epoch: $($retrainSummary.best_epoch)"
        Write-Info "  Best robust accuracy: $($retrainSummary.best_robust_accuracy)"
        Write-Info "  Final clean accuracy: $($retrainSummary.final_clean_accuracy)"
        Write-Info "  Final robust accuracy: $($retrainSummary.final_robust_accuracy)"
        Write-Info ""
    }

    # Show output locations
    Write-Info "Output Locations:"
    Write-Info "  Results directory: $OutputDir"
    Write-Info "  HPO checkpoints: $CheckpointDir"
    Write-Info "  Final model: $FinalCheckpointDir"
    Write-Info "  Analysis plots: $OutputDir/analysis"
    Write-Info "  Logs: logs/"
    Write-Info ""

    Write-Success "Phase 5.4 completed successfully!"
}

function Show-Usage {
    Write-Host @"

USAGE: .\RUN_PHASE_5_4_COMPLETE.ps1 [OPTIONS]

DESCRIPTION:
    Complete automation script for Phase 5.4 TRADES Hyperparameter Optimization.

    The script performs three main steps:
    1. HPO study execution (50 trials with Optuna TPE sampler)
    2. Results analysis and visualization
    3. Model retraining with optimal hyperparameters (200 epochs)

OPTIONS:
    Execution Mode:
        -QuickTest           Run in quick test mode (10 trials, 2 epochs, subset data)
        -SkipHPO             Skip HPO study (only retrain and analyze)
        -SkipRetrain         Skip retraining (only HPO and analysis)
        -SkipAnalysis        Skip analysis and visualization

    HPO Configuration:
        -StudyName <string>  Study name (default: trades_hpo_phase54)
        -NTrials <int>       Number of HPO trials (default: 50)
        -NEpochs <int>       Epochs per trial (default: 10)
        -Storage <string>    Optuna storage URL (default: sqlite:///hpo_study.db)

    Retrain Configuration:
        -RetrainEpochs <int> Epochs for final training (default: 200)
        -UseScheduler        Enable learning rate scheduler
        -UseMLflow           Enable MLflow logging

    Data Configuration:
        -Dataset <string>    Dataset: cifar10, cifar100, svhn (default: cifar10)
        -DataDir <string>    Data directory (default: data)
        -BatchSize <int>     Batch size (default: 128)
        -NumWorkers <int>    Number of workers (default: 4)

    Model Configuration:
        -Model <string>      Model: resnet18, resnet34, resnet50 (default: resnet18)

    Device Configuration:
        -Device <string>     Device: auto, cuda, cpu (default: auto)
        -GpuId <int>         GPU ID to use (default: 0)

    Output Configuration:
        -OutputDir <string>  Output directory (default: results/phase_5_4)
        -CheckpointDir <string>       HPO checkpoint directory
        -FinalCheckpointDir <string>  Final model checkpoint directory

EXAMPLES:
    # Full HPO pipeline (50 trials, 200 epoch retrain)
    .\RUN_PHASE_5_4_COMPLETE.ps1

    # Quick test (10 trials, 2 epochs)
    .\RUN_PHASE_5_4_COMPLETE.ps1 -QuickTest

    # HPO only (skip retraining)
    .\RUN_PHASE_5_4_COMPLETE.ps1 -SkipRetrain

    # Retrain only (skip HPO)
    .\RUN_PHASE_5_4_COMPLETE.ps1 -SkipHPO

    # Full pipeline with scheduler and MLflow
    .\RUN_PHASE_5_4_COMPLETE.ps1 -UseScheduler -UseMLflow

    # Custom configuration
    .\RUN_PHASE_5_4_COMPLETE.ps1 -NTrials 100 -RetrainEpochs 300 -Dataset cifar100

"@ -ForegroundColor $ColorInfo
}

# =============================================================================
# Main Execution
# =============================================================================

function Main {
    Write-Header "PHASE 5.4: TRADES HYPERPARAMETER OPTIMIZATION"

    Write-Info "Configuration:"
    Write-Info "  Quick test mode: $QuickTest"
    Write-Info "  Skip HPO: $SkipHPO"
    Write-Info "  Skip Retrain: $SkipRetrain"
    Write-Info "  Skip Analysis: $SkipAnalysis"
    Write-Info "  Study name: $StudyName"
    Write-Info "  Dataset: $Dataset"
    Write-Info "  Model: $Model"
    Write-Info "  Device: $Device"

    # Check Python environment
    if (-not (Test-PythonEnvironment)) {
        Write-ErrorMsg "Python environment check failed. Please install required packages."
        exit 1
    }

    # Create necessary directories
    Write-Info "Creating output directories..."
    $dirsToCreate = @($OutputDir, $CheckpointDir, $FinalCheckpointDir, "logs")
    foreach ($dir in $dirsToCreate) {
        if (-not (Test-Path $dir)) {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
            Write-Success "Created directory: $dir"
        }
    }

    try {
        # Step 1: HPO Study
        if (-not $SkipHPO) {
            Invoke-HPOStudy
        } else {
            Write-Warning "Skipping HPO study as requested"
        }

        # Step 2: Model Retraining
        if (-not $SkipRetrain) {
            Invoke-ModelRetrain
        } else {
            Write-Warning "Skipping model retraining as requested"
        }

        # Show summary
        Show-Summary

    } catch {
        Write-ErrorMsg "Execution failed: $_"
        Write-ErrorMsg $_.ScriptStackTrace
        exit 1
    }
}

# =============================================================================
# Entry Point
# =============================================================================

# Show help if requested
if ($args -contains "-h" -or $args -contains "--help" -or $args -contains "help") {
    Show-Usage
    exit 0
}

# Run main execution
Main
