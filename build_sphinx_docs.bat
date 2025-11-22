@echo off
REM Build Sphinx documentation for tri-objective-robust-xai-medimg

echo Building Sphinx documentation...

REM Generate API documentation from source code
sphinx-apidoc -f -o docs\api src\ --separate

REM Build HTML documentation
sphinx-build -b html docs docs\_build\html

echo.
echo Documentation built successfully!
echo Open docs\_build\html\index.html to view.

pause
