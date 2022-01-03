try:
    import tensorflow as tf

except:
    raise ModuleNotFoundError("You must install tensorflow for your environment first. This package doesn't contain tensorflow installation.")


from .gpu import get_gpu_objects
from .allocate import allocate
from .gpu import current

__all__ = [
    "get_gpu_objects",
    "allocate",
    "current"
]
