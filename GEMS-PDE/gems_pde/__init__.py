"""GEMS-PDE and the numerically adapted LLM-PDESR baseline."""

import os

for _key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_key] = "1"
__version__ = "0.1.0"
