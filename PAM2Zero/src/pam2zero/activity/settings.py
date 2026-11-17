from __future__ import annotations
import numpy as np
from pam2zero.activity import core
from pam2zero.activity.core import torch, nn
REPO_ROOT = None
DATASETS = list(core.OFFICIAL_SEVEN)
VARIANTS = ["shape_fixed"]
SEEDS = [42, 123, 3407, 2025, 2026]

EPOCHS = 200

BATCH_SIZE = 256

SHAPES = ['EP', 'HelT', 'MGW', 'ProT', 'Roll']

PARAMETERS = {'baseline': 147, 'shape_fixed': 267}

BASE_KEYS = ['guide', 'pam', 'bias', 'interaction', 'edge', 'distance']

LEFT, RIGHT = np.triu_indices(20, 1)
