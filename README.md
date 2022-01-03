# tf_alloc
> Simpliying GPU allocation for Tensorflow
* Using it by pip installation
    pip install tf-alloc



## ⭐️ Why tf_alloc? Problems?
* Compare to pytorch, tensorflow allocate all GPU memory to single training.
* However, it is too much waste because, some training does not use whole GPU memory.
* To solve this problem, TF engineers use two methods.

1. Limit to use only single GPU
2. Limit the use of only a certain percentage of GPUs.

* However, these methods require complex code and memory management.

## ⭐️ Why tf_alloc? How to solve?
#### tf_alloc simplfy and automate GPU allocation using two methods.

## ⭐️ How to allocate?
* Before using tf_alloc, you have to install tensorflow fits for your environment.
* This library does not install specific tensorflow version.

```python
# On the top of the code
from tf_alloc import allocate as talloc
talloc(gpu=1, percentage=0.5)

import tensorflow as tf
""" your code"""
```
#### It is only code for allocating GPU in certain percentage.

### Parameters:
* gpu = which gpu you want to use (if you have two gpu than [0, 1] is possible)
* percentage = the percentage of memory usage on single gpu. 1.0 for maximum use.


## ⭐️ Additional Function.
### GET GPU Objects
    gpu_objs = get_gpu_objects()

* To use this code, you can get gpu objects that contains gpu information.
* You can set GPU backend by using this function.

### GET CURRENT STATE
##### Defualt
    current(
        gpu_id = False, 
        total_memory=False, 
        used = False, 
        free = False, 
        percentage_of_use = False,
        percentage_of_free = False,
    )

* You can use this functions to see current GPU state and possible maximum allocation percentage.
* Without any parameters, than it only visualize possible maximum allocation percentage.
* It is cmd line visualizer. It doesn't return values.

#### Parameters
* gpu_id = visualize the gpu id number
* total_memory = visualize the total memory of GPU
* used = visualize the used memory of GPU
* free = visualize the free memory of GPU
* percentage_of_used = visualize the percentage of used memory of GPU
* percentage_of_free = visualize the percentage of free memory of GPU