# __init__.py
# Copyright (C) 2022 (sta06167@naver.com) and contributors
# Junseo Ko
# github.com/korkite

try:
    import tensorflow as tf

except:
    raise ModuleNotFoundError("You must install tensorflow for your environment first. This package doesn't contain tensorflow installation.")

__version__ = "0.0.1"

from .gpu import get_gpu_objects
from .allocate import allocate
from .gpu import current

__all__ = [
    "get_gpu_objects",
    "allocate",
    "current"
]
