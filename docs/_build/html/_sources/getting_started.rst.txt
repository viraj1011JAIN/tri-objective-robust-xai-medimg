Getting Started
===============

Installation
------------

See the top-level ``README.md`` for full installation instructions. The basic
steps are:

1. Create and activate a Python environment.
2. Install dependencies with Conda or pip.
3. Optionally install DVC and configure a remote for datasets.
4. Run the unit tests to verify the setup.

Quick Test
----------

After installation, you can run the test suite from the project root::

   pytest -q --maxfail=1

For a small tri-objective debug run::

   python scripts/train_tri_objective.py \
     --config configs/experiments/rq1_robustness/tri_objective_debug.yaml \
     --seed 42 \
     --debug

Reproducibility
---------------

The configuration and seed management utilities live in:

- ``src/utils/config.py``
- ``src/utils/reproducibility.py``

These modules provide:

- YAML configuration loading and validation
- Environment variable expansion and path normalisation
- Stable configuration hashing
- Global seed setting and reproducibility state logging
