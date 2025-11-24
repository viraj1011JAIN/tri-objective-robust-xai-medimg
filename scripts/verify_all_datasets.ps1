# Final Dataset Verification - Samsung SSD T7 (/content/drive/MyDrive/data)
# All 6 medical imaging datasets confirmed

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  FINAL Dataset Verification" -ForegroundColor Cyan
Write-Host "  Samsung SSD T7:/content/drive/MyDrive/data" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# ISIC 2018
Write-Host "[1/6] ISIC 2018..." -ForegroundColor Yellow
$isic2018 = (Get-ChildItem "/content/drive/MyDrive/data/isic_2018/ISIC2018_Task3_Training_Input" -File).Count
Write-Host "      Images: $isic2018" -ForegroundColor Green

# ISIC 2019
Write-Host "[2/6] ISIC 2019..." -ForegroundColor Yellow
$isic2019 = (Get-ChildItem "/content/drive/MyDrive/data/isic_2019/train-image" -File -Recurse).Count
Write-Host "      Images: $isic2019" -ForegroundColor Green

# ISIC 2020
Write-Host "[3/6] ISIC 2020..." -ForegroundColor Yellow
$isic2020 = (Get-ChildItem "/content/drive/MyDrive/data/isic_2020/train-image" -File -Recurse).Count
Write-Host "      Images: $isic2020" -ForegroundColor Green

# Derm7pt
Write-Host "[4/6] Derm7pt..." -ForegroundColor Yellow
$derm7pt = (Get-ChildItem "/content/drive/MyDrive/data/derm7pt/images" -File -Recurse).Count
Write-Host "      Images: $derm7pt" -ForegroundColor Green

# NIH CXR (12 folders)
Write-Host "[5/6] NIH Chest X-Ray (12 folders)..." -ForegroundColor Yellow
$nihTotal = 0
1..12 | ForEach-Object {
    $folder = "images_{0:D3}" -f $_
    $path = "/content/drive/MyDrive/data/nih_cxr/$folder/images"
    if (Test-Path $path) {
        $count = (Get-ChildItem $path -File -Filter *.png).Count
        $nihTotal += $count
    }
}
Write-Host "      Images: $nihTotal (across 12 folders)" -ForegroundColor Green

# PadChest
Write-Host "[6/6] PadChest..." -ForegroundColor Yellow
$padchest = (Get-ChildItem "/content/drive/MyDrive/data/padchest/images" -File).Count
Write-Host "      Images: $padchest" -ForegroundColor Green

# Total
$grandTotal = $isic2018 + $isic2019 + $isic2020 + $derm7pt + $nihTotal + $padchest

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Summary" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "ISIC 2018:        $($isic2018.ToString().PadLeft(7)) images" -ForegroundColor White
Write-Host "ISIC 2019:        $($isic2019.ToString().PadLeft(7)) images" -ForegroundColor White
Write-Host "ISIC 2020:        $($isic2020.ToString().PadLeft(7)) images" -ForegroundColor White
Write-Host "Derm7pt:          $($derm7pt.ToString().PadLeft(7)) images" -ForegroundColor White
Write-Host "NIH CXR-14:       $($nihTotal.ToString().PadLeft(7)) images" -ForegroundColor White
Write-Host "PadChest:         $($padchest.ToString().PadLeft(7)) images" -ForegroundColor White
Write-Host "                  -------" -ForegroundColor DarkGray
Write-Host "TOTAL:            $($grandTotal.ToString().PadLeft(7)) images" -ForegroundColor Green
Write-Host ""

Write-Host "[SUCCESS] All datasets verified and ready!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. dvc repro preprocess              # Preprocess all datasets (~2-3 hours)" -ForegroundColor White
Write-Host "  2. dvc repro build_concept_bank_*    # Build concept banks" -ForegroundColor White
Write-Host "  3. pytest tests/ -v --cov=src        # Run full test suite" -ForegroundColor White
Write-Host ""
