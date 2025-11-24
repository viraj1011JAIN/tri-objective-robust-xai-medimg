# Dataset Verification Script for Samsung SSD T7 (/content/drive/MyDrive/data)
# Author: Viraj Pankaj Jain
# Purpose: Verify all medical imaging datasets are properly downloaded

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Dataset Verification -/content/drive/MyDrive/data" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$datasets = @(
    @{Name="ISIC 2018"; Path="/content/drive/MyDrive/data/isic_2018"; Metadata="metadata.csv"; Images="ISIC2018_Task3_Training_Input"},
    @{Name="ISIC 2019"; Path="/content/drive/MyDrive/data/isic_2019"; Metadata="metadata.csv"; Images="train-image"},
    @{Name="ISIC 2020"; Path="/content/drive/MyDrive/data/isic_2020"; Metadata="metadata.csv"; Images="train-image"},
    @{Name="Derm7pt"; Path="/content/drive/MyDrive/data/derm7pt"; Metadata="metadata.csv"; Images="images"},
    @{Name="NIH CXR"; Path="/content/drive/MyDrive/data/nih_cxr"; Metadata="Data_Entry_2017.csv"; Images="images_001"},
    @{Name="PadChest"; Path="/content/drive/MyDrive/data/padchest"; Metadata="metadata.csv"; Images="images"}
)

$allPassed = $true

foreach ($ds in $datasets) {
    Write-Host ""
    Write-Host "Checking: $($ds.Name)" -ForegroundColor Yellow
    Write-Host "----------------------------------------" -ForegroundColor Yellow

    # Check main path
    if (Test-Path $ds.Path) {
        Write-Host "[OK] Dataset path exists: $($ds.Path)" -ForegroundColor Green
    } else {
        Write-Host "[FAIL] Dataset path NOT found: $($ds.Path)" -ForegroundColor Red
        $allPassed = $false
        continue
    }

    # Check metadata
    $metadataPath = Join-Path $ds.Path $ds.Metadata
    if (Test-Path $metadataPath) {
        Write-Host "[OK] Metadata exists: $($ds.Metadata)" -ForegroundColor Green
    } else {
        Write-Host "[FAIL] Metadata NOT found: $($ds.Metadata)" -ForegroundColor Red
        $allPassed = $false
    }

    # Check images folder
    $imagesPath = Join-Path $ds.Path $ds.Images
    if (Test-Path $imagesPath) {
        Write-Host "[OK] Images folder exists: $($ds.Images)" -ForegroundColor Green
        $imageCount = (Get-ChildItem -Path $imagesPath -File -Recurse -Include *.jpg,*.jpeg,*.png -ErrorAction SilentlyContinue).Count
        Write-Host "     Image count: $imageCount" -ForegroundColor Cyan
    } else {
        Write-Host "[FAIL] Images folder NOT found: $($ds.Images)" -ForegroundColor Red
        $allPassed = $false
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Verification Results" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

if ($allPassed) {
    Write-Host "[SUCCESS] All datasets verified!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor White
    Write-Host "  1. Run preprocessing: dvc repro preprocess" -ForegroundColor Cyan
    Write-Host "  2. Build concept banks: dvc repro build_concept_bank_isic2018" -ForegroundColor Cyan
    Write-Host "  3. Run tests: pytest tests/ -v" -ForegroundColor Cyan
} else {
    Write-Host "[ERROR] Some datasets are incomplete!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please check missing files and re-run this script." -ForegroundColor Yellow
}

Write-Host ""
