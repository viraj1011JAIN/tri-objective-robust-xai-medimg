@echo off
REM Training Script Launcher with Correct Python
REM This script ensures we use Python 3.11 which has PyTorch installed

"C:\Users\Viraj Jain\AppData\Local\Programs\Python\Python311\python.exe" -m src.training.train_baseline %*
