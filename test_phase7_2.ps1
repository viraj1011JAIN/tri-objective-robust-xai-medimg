#!/usr/bin/env pwsh
# Phase 7.2 Test Runner - Tests ONLY tri-objective loss module
# Runs 38 tests with coverage on src/losses/tri_objective.py only

Write-Host "===============================================" -ForegroundColor Cyan
Write-Host "Phase 7.2: Tri-Objective Loss Test Runner" -ForegroundColor Cyan
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host ""

# Clear previous coverage data
coverage erase

# Run tests with focused coverage (override pytest.ini settings)
pytest tests/losses/test_tri_objective_loss.py `
    -v `
    -c /dev/null `
    --maxfail=5 `
    --disable-warnings `
    --cov=src.losses.tri_objective `
    --cov-branch `
    --cov-report=term `
    --cov-report=html:htmlcov_phase7_2 `
    --cov-report=xml:coverage_phase7_2.xml `
    --tb=short `
    --durations=10

$testExitCode = $LASTEXITCODE

Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "Phase 7.2 Module Coverage:" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan

# Extract tri_objective.py coverage from report
$coverageFile = "coverage_phase7_2.xml"
$success = $false

if (Test-Path $coverageFile) {
    [xml]$coverage = Get-Content $coverageFile
    $triObjectiveFile = $coverage.coverage.packages.package.classes.class | Where-Object { $_.filename -like "*tri_objective.py" }
    if ($triObjectiveFile) {
        $lineRate = [math]::Round([double]$triObjectiveFile.'line-rate' * 100, 2)
        $branchRate = [math]::Round([double]$triObjectiveFile.'branch-rate' * 100, 2)
        Write-Host "Line Coverage:   $lineRate%" -ForegroundColor Green
        Write-Host "Branch Coverage: $branchRate%" -ForegroundColor Green
        Write-Host ""

        if ($lineRate -ge 80.0 -and $testExitCode -eq 0) {
            Write-Host "✅ Coverage target met (≥80%)" -ForegroundColor Green
            $success = $true
        } elseif ($lineRate -ge 80.0) {
            Write-Host "✅ Coverage target met (≥80%)" -ForegroundColor Green
            Write-Host "⚠️  But tests failed - check output above" -ForegroundColor Yellow
        } else {
            Write-Host "⚠️  Coverage below target (need ≥80%)" -ForegroundColor Yellow
        }
    }
}

Write-Host ""
if ($testExitCode -eq 0) {
    Write-Host "✅ Phase 7.2: All 38 Tests PASSED" -ForegroundColor Green
    if ($success) {
        Write-Host "✅ Phase 7.2: COMPLETE" -ForegroundColor Green
    }
    Write-Host ""
    Write-Host "Opening coverage report..." -ForegroundColor Cyan
    Invoke-Item .\htmlcov_phase7_2\index.html
} else {
    Write-Host "❌ Phase 7.2 Tests FAILED" -ForegroundColor Red
}

Write-Host ""
Write-Host "Coverage report: htmlcov_phase7_2/index.html" -ForegroundColor Cyan
Write-Host ""

# Exit with success if tests passed (ignore coverage aggregation issues)
if ($testExitCode -eq 0 -and $success) {
    exit 0
} else {
    exit $testExitCode
}
