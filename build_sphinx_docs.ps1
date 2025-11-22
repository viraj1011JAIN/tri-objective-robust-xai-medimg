# Build Sphinx Documentation
# PowerShell script for generating API documentation

Write-Host "Building Sphinx documentation..." -ForegroundColor Green

# Generate API documentation from source code
Write-Host "`nGenerating API documentation from source..." -ForegroundColor Cyan
sphinx-apidoc -f -o docs/api src/ --separate

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error generating API documentation" -ForegroundColor Red
    exit 1
}

# Build HTML documentation
Write-Host "`nBuilding HTML documentation..." -ForegroundColor Cyan
sphinx-build -b html docs docs/_build/html

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error building HTML documentation" -ForegroundColor Red
    exit 1
}

Write-Host "`nDocumentation built successfully!" -ForegroundColor Green
Write-Host "Open docs/_build/html/index.html to view" -ForegroundColor Yellow
