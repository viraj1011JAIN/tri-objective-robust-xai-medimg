# Tri-Objective Robust XAI for Medical Imaging

This repository implements adversarially robust and concept-grounded explainable deep learning for medical imaging
(CNNs and Vision Transformers on NIH CXR, ISIC, Derm7pt, and PadChest).

## Environment setup

### Option 1: Conda (local machine)

```bash
conda env create -f environment.yml
conda activate triobj-medimg
python scripts/check_env.py
```

### Option 2: Pip (local machine)

```bash
pip install -r requirements.txt
python scripts/check_env.py
```

### Option 3: Docker (local or cluster GPU machine)

Build image
```bash
docker build -t triobj-medimg:latest .
```

Run container with GPU support
```bash
docker run --gpus all -it --rm -v $PWD:/workspace triobj-medimg:latest
```

Inside the container, verify the environment:
```bash
python scripts/check_env.py
```

This prints the Python and library versions and shows whether CUDA is available.
