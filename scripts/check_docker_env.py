from __future__ import annotations

import json
import os
import platform
from typing import Optional

import torch


def try_import_any(candidates: list[str]) -> tuple[bool, Optional[str], str]:
    """
    Try importing a list of possible package/module names.

    Returns:
        (ok, imported_name, error_message)
    """
    last_error = ""
    for name in candidates:
        try:
            __import__(name)
            return True, name, ""
        except Exception as e:  # noqa: BLE001 - we want the full repr
            last_error = repr(e)
    return False, None, last_error


def main() -> None:
    # Which modules to test? First the “nice” package name, then the real code root.
    candidates = [
        "tri_objective_robust_xai_medimg",  # distribution name (if ever packaged)
        "src",  # your actual source root
        "src.train",  # a concrete module in the project
    ]

    package_import_ok, imported_name, package_import_error = try_import_any(candidates)

    cuda_available = torch.cuda.is_available()
    if cuda_available:
        cuda_device_count = torch.cuda.device_count()
        cuda_device_name_0 = (
            torch.cuda.get_device_name(0) if cuda_device_count > 0 else None
        )
    else:
        cuda_device_count = 0
        cuda_device_name_0 = None

    info = {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_device_count": cuda_device_count,
        "cuda_device_name_0": cuda_device_name_0,
        "platform": platform.platform(),
        "working_dir": os.getcwd(),
        "package_import_ok": package_import_ok,
        "package_import_name": imported_name,
        "package_import_error": package_import_error,
    }

    print("=== Docker environment check ===")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
