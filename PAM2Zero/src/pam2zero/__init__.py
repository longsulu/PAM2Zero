import os

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

__version__ = "0.1.0"
