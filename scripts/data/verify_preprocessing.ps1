# scripts/data/verify_preprocessing.ps1
# Automated verification script for Phase 2.5 preprocessing

$datasets = @("isic2018", "isic2019", "isic2020", "derm7pt", "nih_cxr", "padchest")
$results = @()

Write-Host "`n===================================================================" -ForegroundColor Cyan
Write-Host "Phase 2.5 Preprocessing Verification Report" -ForegroundColor Cyan
Write-Host "===================================================================`n" -ForegroundColor Cyan

foreach ($ds in $datasets) {
    Write-Host "Verifying $ds..." -ForegroundColor Yellow

    $result = [PSCustomObject]@{
        Dataset = $ds
        DirectoryExists = $false
        LogExists = $false
        MetadataExists = $false
        HDF5Exists = $false
        DVCTracked = $false
        TotalSamples = 0
        ProcessingTime = 0
        Status = "NOT_STARTED"
    }

    # Check directory
    $dir = "data\processed\$ds"
    if (Test-Path $dir) {
        $result.DirectoryExists = $true
        Write-Host "  ✅ Directory exists" -ForegroundColor Green

        # Check log
        $logPath = "$dir\preprocess_log.json"
        if (Test-Path $logPath) {
            $result.LogExists = $true
            try {
                $logData = Get-Content $logPath | ConvertFrom-Json
                $result.TotalSamples = $logData.num_processed_rows
                $result.ProcessingTime = [math]::Round($logData.run_time_sec, 2)
                Write-Host "  ✅ Log exists: $($result.TotalSamples) samples in $($result.ProcessingTime)s" -ForegroundColor Green
            } catch {
                Write-Host "  ⚠️  Log file corrupted" -ForegroundColor Yellow
            }
        } else {
            Write-Host "  ❌ Log missing" -ForegroundColor Red
        }

        # Check metadata
        $metadataPath = "$dir\metadata_processed.csv"
        if (Test-Path $metadataPath) {
            $result.MetadataExists = $true
            Write-Host "  ✅ Metadata CSV exists" -ForegroundColor Green
        } else {
            Write-Host "  ❌ Metadata CSV missing" -ForegroundColor Red
        }

        # Check HDF5
        $hdf5Path = "$dir\dataset.h5"
        if (Test-Path $hdf5Path) {
            $result.HDF5Exists = $true
            $hdf5Size = [math]::Round((Get-Item $hdf5Path).Length / 1GB, 2)
            Write-Host "  ✅ HDF5 file exists ($hdf5Size GB)" -ForegroundColor Green
        } else {
            Write-Host "  ⚠️  HDF5 file not found" -ForegroundColor Yellow
        }

        # Check DVC tracking
        if (Test-Path "$dir.dvc") {
            $result.DVCTracked = $true
            Write-Host "  ✅ DVC tracked" -ForegroundColor Green
        } else {
            Write-Host "  ⚠️  Not DVC tracked yet" -ForegroundColor Yellow
        }

        # Determine status
        if ($result.LogExists -and $result.MetadataExists) {
            $result.Status = "COMPLETE"
        } else {
            $result.Status = "INCOMPLETE"
        }
    } else {
        Write-Host "  ❌ Directory not found" -ForegroundColor Red
        $result.Status = "NOT_STARTED"
    }

    $results += $result
    Write-Host ""
}

# Summary Table
Write-Host "`n===================================================================" -ForegroundColor Cyan
Write-Host "Summary" -ForegroundColor Cyan
Write-Host "===================================================================`n" -ForegroundColor Cyan

$results | Format-Table -AutoSize Dataset, Status, TotalSamples, ProcessingTime, HDF5Exists, DVCTracked

# Overall Status
$completed = ($results | Where-Object { $_.Status -eq "COMPLETE" }).Count
$total = $datasets.Count

Write-Host "`nOverall Progress: $completed / $total datasets completed" -ForegroundColor $(if ($completed -eq $total) { "Green" } else { "Yellow" })

# Save report to JSON
$reportPath = "docs\reports\preprocessing_verification_report.json"
$results | ConvertTo-Json -Depth 10 | Out-File $reportPath -Encoding UTF8
Write-Host "`nDetailed report saved to: $reportPath" -ForegroundColor Cyan

# Exit code based on completion
if ($completed -eq $total) {
    Write-Host "`n✅ All datasets preprocessed successfully!" -ForegroundColor Green
    exit 0
} else {
    Write-Host "`n⚠️  Some datasets still need preprocessing" -ForegroundColor Yellow
    exit 1
}
