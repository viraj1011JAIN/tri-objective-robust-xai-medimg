# ==============================================================================
# Phase 5.3 - Complete TRADES Training & Evaluation Automation
# ==============================================================================
# This script automates the entire TRADES pipeline:
#   1. Training TRADES models (3 seeds × 3 architectures = 9 models)
#   2. Evaluation on test set + adversarial attacks
#   3. Comparison with Phase 5.2 (PGD-AT) baseline
#   4. Trade-off analysis and Pareto frontier visualization
#   5. Results aggregation and reporting
#
# Author: Viraj Pankaj Jain
# Institution: University of Glasgow
# Date: November 2025
# ==============================================================================

param(
    [switch]$SkipTraining,
    [switch]$SkipEvaluation,
    [switch]$SkipComparison,
    [switch]$SkipVisualization
)

# Configuration
$ErrorActionPreference = "Stop"
$ProgressPreference = "Continue"

$SEEDS = @(42, 123, 456)
$MODELS = @("resnet50", "efficientnet_b0", "vit_b_16")
$CONFIG = "configs/experiments/trades_isic.yaml"
$PHASE_5_2_RESULTS = "results/phase_5_2_pgd_at/evaluation_metrics"
$OUTPUT_BASE = "results/phase_5_3_trades"

# Color output functions
function Write-Header {
    param([string]$Message)
    Write-Host "`n==============================================================================`n" -ForegroundColor Cyan
    Write-Host "  $Message" -ForegroundColor Cyan
    Write-Host "`n==============================================================================`n" -ForegroundColor Cyan
}

function Write-Success {
    param([string]$Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor Green
}

function Write-Info {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Blue
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

# Create output directories
function Initialize-Directories {
    Write-Header "Initializing Output Directories"

    $dirs = @(
        "$OUTPUT_BASE",
        "$OUTPUT_BASE/checkpoints",
        "$OUTPUT_BASE/logs",
        "$OUTPUT_BASE/evaluation_metrics",
        "$OUTPUT_BASE/evaluation_plots",
        "$OUTPUT_BASE/comparison",
        "$OUTPUT_BASE/reports"
    )

    foreach ($dir in $dirs) {
        if (-not (Test-Path $dir)) {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
            Write-Info "Created directory: $dir"
        }
    }

    Write-Success "Directories initialized"
}

# Train TRADES models
function Start-Training {
    Write-Header "Phase 5.3 - TRADES Training"

    $total = $SEEDS.Count * $MODELS.Count
    $current = 0

    foreach ($seed in $SEEDS) {
        foreach ($model in $MODELS) {
            $current++
            Write-Info "[$current/$total] Training $model with seed $seed"

            $startTime = Get-Date

            try {
                python scripts/training/train_trades.py `
                    --config $CONFIG `
                    --seed $seed `
                    --model $model

                $endTime = Get-Date
                $duration = ($endTime - $startTime).TotalMinutes

                Write-Success "Completed $model (seed=$seed) in $([math]::Round($duration, 2)) minutes"
            }
            catch {
                Write-Error "Training failed for $model (seed=$seed): $_"
                throw
            }
        }
    }

    Write-Success "All training runs completed"
}

# Evaluate TRADES models
function Start-Evaluation {
    Write-Header "Phase 5.3 - TRADES Evaluation"

    $total = $SEEDS.Count * $MODELS.Count
    $current = 0

    foreach ($seed in $SEEDS) {
        foreach ($model in $MODELS) {
            $current++
            Write-Info "[$current/$total] Evaluating $model with seed $seed"

            $checkpoint = "$OUTPUT_BASE/checkpoints/${model}_seed_${seed}/best.pt"

            if (-not (Test-Path $checkpoint)) {
                Write-Warning "Checkpoint not found: $checkpoint"
                continue
            }

            $outputDir = "$OUTPUT_BASE/evaluation_metrics/${model}_seed_${seed}"

            try {
                python scripts/evaluation/evaluate_trades.py `
                    --config $CONFIG `
                    --checkpoint $checkpoint `
                    --output_dir $outputDir

                Write-Success "Completed evaluation for $model (seed=$seed)"
            }
            catch {
                Write-Error "Evaluation failed for $model (seed=$seed): $_"
                throw
            }
        }
    }

    Write-Success "All evaluations completed"
}

# Compare TRADES vs PGD-AT
function Start-Comparison {
    Write-Header "Phase 5.3 - TRADES vs PGD-AT Comparison"

    if (-not (Test-Path $PHASE_5_2_RESULTS)) {
        Write-Error "Phase 5.2 results not found at: $PHASE_5_2_RESULTS"
        Write-Error "Please complete Phase 5.2 training first or update the path"
        throw
    }

    Write-Info "Comparing TRADES with PGD-AT baseline..."

    try {
        # Aggregate results
        Write-Info "Aggregating TRADES results..."
        python scripts/analysis/aggregate_results.py `
            --results_dir "$OUTPUT_BASE/evaluation_metrics" `
            --output "$OUTPUT_BASE/comparison/trades_aggregated.json"

        Write-Info "Aggregating PGD-AT results..."
        python scripts/analysis/aggregate_results.py `
            --results_dir $PHASE_5_2_RESULTS `
            --output "$OUTPUT_BASE/comparison/pgd_at_aggregated.json"

        # Statistical comparison
        Write-Info "Performing statistical tests..."
        python src/evaluation/comparison.py `
            --trades_results "$OUTPUT_BASE/comparison/trades_aggregated.json" `
            --baseline_results "$OUTPUT_BASE/comparison/pgd_at_aggregated.json" `
            --output "$OUTPUT_BASE/comparison/statistical_comparison.json" `
            --seeds 42 123 456

        Write-Success "Comparison completed"
    }
    catch {
        Write-Error "Comparison failed: $_"
        throw
    }
}

# Generate visualizations
function Start-Visualization {
    Write-Header "Phase 5.3 - Visualization Generation"

    Write-Info "Performing trade-off analysis..."

    try {
        # Trade-off analysis
        python src/evaluation/tradeoff_analysis.py `
            --results_dir "$OUTPUT_BASE/evaluation_metrics" `
            --output "$OUTPUT_BASE/comparison/tradeoff_analysis.json"

        # Pareto curve visualization
        Write-Info "Generating Pareto frontier plots..."
        python src/visualization/pareto_curves.py `
            --results "$OUTPUT_BASE/comparison/tradeoff_analysis.json" `
            --output_dir "$OUTPUT_BASE/evaluation_plots"

        # Generate summary plots
        Write-Info "Generating summary visualizations..."
        python scripts/analysis/create_summary_plots.py `
            --trades_results "$OUTPUT_BASE/comparison/trades_aggregated.json" `
            --baseline_results "$OUTPUT_BASE/comparison/pgd_at_aggregated.json" `
            --output_dir "$OUTPUT_BASE/evaluation_plots"

        Write-Success "Visualizations generated"
    }
    catch {
        Write-Error "Visualization failed: $_"
        throw
    }
}

# Generate final report
function New-Report {
    Write-Header "Generating Final Report"

    $reportPath = "$OUTPUT_BASE/reports/PHASE_5_3_FINAL_REPORT.md"

    try {
        python scripts/analysis/generate_report.py `
            --comparison "$OUTPUT_BASE/comparison/statistical_comparison.json" `
            --tradeoff "$OUTPUT_BASE/comparison/tradeoff_analysis.json" `
            --output $reportPath

        Write-Success "Report generated: $reportPath"
    }
    catch {
        Write-Error "Report generation failed: $_"
        throw
    }
}

# Main execution
function Main {
    $startTime = Get-Date

    Write-Header "Phase 5.3 - TRADES Complete Pipeline"
    Write-Info "Start time: $startTime"

    # Initialize
    Initialize-Directories

    # Training
    if (-not $SkipTraining) {
        Start-Training
    }
    else {
        Write-Warning "Skipping training (--SkipTraining flag set)"
    }

    # Evaluation
    if (-not $SkipEvaluation) {
        Start-Evaluation
    }
    else {
        Write-Warning "Skipping evaluation (--SkipEvaluation flag set)"
    }

    # Comparison
    if (-not $SkipComparison) {
        Start-Comparison
    }
    else {
        Write-Warning "Skipping comparison (--SkipComparison flag set)"
    }

    # Visualization
    if (-not $SkipVisualization) {
        Start-Visualization
    }
    else {
        Write-Warning "Skipping visualization (--SkipVisualization flag set)"
    }

    # Generate report
    New-Report

    $endTime = Get-Date
    $totalDuration = ($endTime - $startTime).TotalMinutes

    Write-Header "Phase 5.3 - Pipeline Complete"
    Write-Success "Total execution time: $([math]::Round($totalDuration, 2)) minutes"
    Write-Info "Results saved to: $OUTPUT_BASE"
    Write-Info "Report available at: $OUTPUT_BASE/reports/PHASE_5_3_FINAL_REPORT.md"
}

# Run
try {
    Main
}
catch {
    Write-Error "Pipeline failed: $_"
    exit 1
}
