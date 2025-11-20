# Training Script Launcher with Correct Python
# This script ensures we use Python 3.11 which has PyTorch installed

$python311 = "C:\Users\Viraj Jain\AppData\Local\Programs\Python\Python311\python.exe"

& $python311 -m src.training.train_baseline $args
