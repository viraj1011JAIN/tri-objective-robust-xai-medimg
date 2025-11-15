import sys

print("Python:", sys.version)

modules = [
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("torchaudio", "torchaudio"),
    ("timm", "timm"),
    ("captum", "captum"),
    ("mlflow", "mlflow"),
    ("dvc", "dvc"),
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("sklearn", "sklearn"),
    ("pandas", "pandas"),
    ("matplotlib", "matplotlib"),
    ("seaborn", "seaborn"),
    ("PIL", "PIL"),
    ("cv2", "cv2"),
    ("albumentations", "albumentations"),
    ("einops", "einops"),
    ("yaml", "yaml"),
    ("tqdm", "tqdm"),
    ("rich", "rich"),
]

for name, import_name in modules:
    try:
        mod = __import__(import_name)
        version = getattr(mod, "__version__", "N/A")
        print(f"{name}: {version}")
    except Exception as e:
        print(f"{name}: ERROR ({e})")

try:
    import torch

    print("CUDA available:", torch.cuda.is_available())
except ImportError:
    print("torch: NOT INSTALLED")
