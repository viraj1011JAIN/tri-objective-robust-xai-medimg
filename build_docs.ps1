# Build Documentation Script
# Builds Sphinx HTML documentation and optionally opens it

param(
    [switch]$Open,
    [switch]$Clean
)

$PYTHON311 = "C:\Users\Viraj Jain\AppData\Local\Programs\Python\Python311\python.exe"

Write-Host "`n" -NoNewline
Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host "Building Sphinx Documentation" -ForegroundColor Cyan
Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host ""

# Clean build directory if requested
if ($Clean) {
    Write-Host "[1/3] Cleaning old build..." -ForegroundColor Yellow
    Push-Location docs
    & $PYTHON311 -m sphinx -M clean . _build
    Pop-Location
    Write-Host "✓ Build directory cleaned`n" -ForegroundColor Green
}

# Build documentation
Write-Host "[2/3] Building HTML documentation..." -ForegroundColor Yellow
Push-Location docs
& $PYTHON311 -m sphinx -b html . _build/html
$buildSuccess = $LASTEXITCODE -eq 0
Pop-Location

if ($buildSuccess) {
    Write-Host "`n✓ Documentation built successfully!`n" -ForegroundColor Green

    # Show output location
    Write-Host "📁 Output Location:" -ForegroundColor Cyan
    Write-Host "   docs\_build\html\index.html`n"

    # Open in browser if requested
    if ($Open) {
        Write-Host "[3/3] Opening documentation in browser..." -ForegroundColor Yellow
        Start-Process docs\_build\html\index.html
        Write-Host "✓ Documentation opened`n" -ForegroundColor Green
    }

    Write-Host "=" * 70 -ForegroundColor Cyan
    Write-Host "Build Complete!" -ForegroundColor Green
    Write-Host "=" * 70 -ForegroundColor Cyan
} else {
    Write-Host "`n✗ Build failed with errors`n" -ForegroundColor Red
    Write-Host "Check the output above for details.`n" -ForegroundColor Yellow
    exit 1
}

Write-Host "`nUsage Examples:" -ForegroundColor Cyan
Write-Host "  .\build_docs.ps1              # Build documentation"
Write-Host "  .\build_docs.ps1 -Open        # Build and open in browser"
Write-Host "  .\build_docs.ps1 -Clean       # Clean and rebuild"
Write-Host "  .\build_docs.ps1 -Clean -Open # Clean, rebuild, and open`n"
