# ==============================================================================
# Dataset Preparation and Verification Script
# ==============================================================================
# Purpose: Verify and prepare all datasets from Samsung SSD T7 (D:/data)
# Author: Viraj Pankaj Jain
# Date: November 23, 2025
# ==============================================================================

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "   Dataset Preparation & Verification" -ForegroundColor Cyan
Write-Host "   Samsung SSD T7: D:/data" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Dataset definitions
$datasets = @(
    @{
        Name = "ISIC 2018"
        Path = "D:/data/isic_2018"
        MetadataFile = "metadata.csv"
        ImageFolder = "ISIC2018_Task3_Training_Input"
        GroundTruthFile = "ISIC2018_Task3_Training_GroundTruth.csv"
        ExpectedImages = 10015
        Classes = 7
    },
    @{
        Name = "ISIC 2019"
        Path = "D:/data/isic_2019"
        MetadataFile = "metadata.csv"
        ImageFolder = "ISIC_2019_Training_Input"
        GroundTruthFile = "ISIC_2019_Training_GroundTruth.csv"
        ExpectedImages = 25331
        Classes = 8
    },
    @{
        Name = "ISIC 2020"
        Path = "D:/data/isic_2020"
        MetadataFile = "metadata.csv"
        ImageFolder = "train"
        GroundTruthFile = "train.csv"
        ExpectedImages = 33126
        Classes = 2
    },
    @{
        Name = "Derm7pt"
        Path = "D:/data/derm7pt"
        MetadataFile = "metadata.csv"
        ImageFolder = "images"
        GroundTruthFile = "meta/meta.csv"
        ExpectedImages = 2000
        Classes = 2
    },
    @{
        Name = "NIH Chest X-Ray"
        Path = "D:/data/nih_cxr"
        MetadataFile = "Data_Entry_2017.csv"
        ImageFolder = "images"
        GroundTruthFile = "Data_Entry_2017.csv"
        ExpectedImages = 112120
        Classes = 14
    },
    @{
        Name = "PadChest"
        Path = "D:/data/padchest"
        MetadataFile = "PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv"
        ImageFolder = "images"
        GroundTruthFile = "PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv"
        ExpectedImages = 39000
        Classes = 174
    }
)

$allPassed = $true
$verificationResults = @()

# Function to check dataset
function Test-Dataset {
    param($dataset)

    Write-Host ""
    Write-Host "----------------------------------------" -ForegroundColor Yellow
    Write-Host "Checking: $($dataset.Name)" -ForegroundColor Yellow
    Write-Host "----------------------------------------" -ForegroundColor Yellow

    $result = @{
        Name = $dataset.Name
        PathExists = $false
        MetadataExists = $false
        ImageFolderExists = $false
        GroundTruthExists = $false
        Status = "FAIL"
        Issues = @()
    }

    # Check main path
    if (Test-Path $dataset.Path) {
        Write-Host "✅ Dataset path exists: $($dataset.Path)" -ForegroundColor Green
        $result.PathExists = $true
    } else {
        Write-Host "❌ Dataset path NOT found: $($dataset.Path)" -ForegroundColor Red
        $result.Issues += "Dataset path missing"
        return $result
    }

    # Check metadata file
    $metadataPath = Join-Path $dataset.Path $dataset.MetadataFile
    if (Test-Path $metadataPath) {
        Write-Host "✅ Metadata file exists: $($dataset.MetadataFile)" -ForegroundColor Green
        $result.MetadataExists = $true

        # Try to count rows
        try {
            $csvData = Import-Csv $metadataPath
            $rowCount = $csvData.Count
            Write-Host "   📊 Metadata rows: $rowCount" -ForegroundColor Cyan
        } catch {
            Write-Host "   ⚠️  Could not read metadata CSV" -ForegroundColor Yellow
        }
    } else {
        Write-Host "❌ Metadata file NOT found: $($dataset.MetadataFile)" -ForegroundColor Red
        $result.Issues += "Metadata file missing"
    }

    # Check image folder
    $imageFolderPath = Join-Path $dataset.Path $dataset.ImageFolder
    if (Test-Path $imageFolderPath) {
        Write-Host "✅ Image folder exists: $($dataset.ImageFolder)" -ForegroundColor Green
        $result.ImageFolderExists = $true

        # Count images
        $imageCount = (Get-ChildItem -Path $imageFolderPath -File -Recurse -Include *.jpg,*.jpeg,*.png -ErrorAction SilentlyContinue).Count
        Write-Host "   🖼️  Images found: $imageCount" -ForegroundColor Cyan

        if ($imageCount -lt ($dataset.ExpectedImages * 0.9)) {
            Write-Host "   ⚠️  Image count below expected ($($dataset.ExpectedImages))" -ForegroundColor Yellow
            $result.Issues += "Low image count"
        }
    } else {
        Write-Host "❌ Image folder NOT found: $($dataset.ImageFolder)" -ForegroundColor Red
        $result.Issues += "Image folder missing"
    }

    # Check ground truth file
    $groundTruthPath = Join-Path $dataset.Path $dataset.GroundTruthFile
    if (Test-Path $groundTruthPath) {
        Write-Host "✅ Ground truth file exists: $($dataset.GroundTruthFile)" -ForegroundColor Green
        $result.GroundTruthExists = $true
    } else {
        Write-Host "⚠️  Ground truth file not found: $($dataset.GroundTruthFile)" -ForegroundColor Yellow
        # Not critical if metadata exists
    }

    # Determine overall status
    if ($result.PathExists -and $result.MetadataExists -and $result.ImageFolderExists) {
        $result.Status = "PASS"
        Write-Host ""
        Write-Host "✅ $($dataset.Name): READY" -ForegroundColor Green
    } else {
        $result.Status = "FAIL"
        Write-Host ""
        Write-Host "❌ $($dataset.Name): NOT READY" -ForegroundColor Red
        $script:allPassed = $false
    }

    return $result
}

# Check all datasets
foreach ($dataset in $datasets) {
    $result = Test-Dataset -dataset $dataset
    $verificationResults += $result
}

# Summary
Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "   Verification Summary" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

$passCount = ($verificationResults | Where-Object { $_.Status -eq "PASS" }).Count
$totalCount = $verificationResults.Count

Write-Host "Datasets checked: $totalCount" -ForegroundColor White
Write-Host "Passed: $passCount" -ForegroundColor Green
Write-Host "Failed: $($totalCount - $passCount)" -ForegroundColor Red

Write-Host ""
Write-Host "--- Detailed Results ---" -ForegroundColor Cyan
Write-Host ""

foreach ($result in $verificationResults) {
    $statusColor = if ($result.Status -eq "PASS") { "Green" } else { "Red" }
    $statusIcon = if ($result.Status -eq "PASS") { "✅" } else { "❌" }

    Write-Host "$statusIcon $($result.Name): $($result.Status)" -ForegroundColor $statusColor

    if ($result.Issues.Count -gt 0) {
        foreach ($issue in $result.Issues) {
            Write-Host "   ⚠️  $issue" -ForegroundColor Yellow
        }
    }
}

# Next steps
Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "   Next Steps" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

if ($allPassed) {
    Write-Host "✅ All datasets verified and ready!" -ForegroundColor Green
    Write-Host ""
    Write-Host "You can now:" -ForegroundColor White
    Write-Host "  1. Run preprocessing: dvc repro preprocess" -ForegroundColor Cyan
    Write-Host "  2. Build concept banks: dvc repro build_concept_bank_isic2018" -ForegroundColor Cyan
    Write-Host "  3. Run tests with real data: pytest tests/ -v" -ForegroundColor Cyan
} else {
    Write-Host "❌ Some datasets are not ready." -ForegroundColor Red
    Write-Host ""
    Write-Host "Please:" -ForegroundColor White
    Write-Host "  1. Check missing files/folders" -ForegroundColor Yellow
    Write-Host "  2. Verify dataset downloads are complete" -ForegroundColor Yellow
    Write-Host "  3. Re-run this script after fixing issues" -ForegroundColor Yellow
}

# Check processed directories
Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "   Processed Data Directories" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

$processedDir = "C:\Users\Dissertation\tri-objective-robust-xai-medimg\data\processed"
if (Test-Path $processedDir) {
    Write-Host "✅ Processed directory exists: $processedDir" -ForegroundColor Green

    $subdirs = Get-ChildItem -Path $processedDir -Directory -ErrorAction SilentlyContinue
    if ($subdirs) {
        Write-Host ""
        Write-Host "Existing processed datasets:" -ForegroundColor Cyan
        foreach ($subdir in $subdirs) {
            Write-Host "  • $($subdir.Name)" -ForegroundColor White
        }
    } else {
        Write-Host "  (No processed datasets yet - will be created during preprocessing)" -ForegroundColor Yellow
    }
} else {
    Write-Host "⚠️  Processed directory will be created: $processedDir" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "   Verification Complete" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
