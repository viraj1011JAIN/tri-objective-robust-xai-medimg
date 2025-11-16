from __future__ import annotations

import os
import sys
from datetime import datetime

# Add project root and src/ to sys.path so autodoc can find modules
PROJECT_ROOT = os.path.abspath("..")
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SRC_ROOT)

project = "Tri-Objective Robust XAI for Medical Imaging"
author = "Viraj Pankaj Jain"
year = datetime.now().year
copyright = f"{year}, {author}"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
]

autosummary_generate = True

templates_path = ["_templates"]
exclude_patterns: list[str] = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"

# Napoleon settings for Google / NumPy style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True
