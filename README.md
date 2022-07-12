# tf_alloc
> Simpliying GPU allocation for Tensorflow
* Developer: korkite (Junseo Ko)

## Installation  
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
from tf_alloc import allocate
allocate(gpu=1, percentage=0.5)

import tensorflow as tf
""" your code"""
```
#### It is only code for allocating GPU in certain percentage.

### Parameters:
* gpu = which gpu you want to use (if you have two gpu than [0, 1] is possible)
* percentage = the percentage of memory usage on single gpu. 1.0 for maximum use.


## ⭐️ Additional Function.
### GET GPU Objects
```python3
from tf_alloc import get_gpu_objects
gpu_objs = get_gpu_objects()
```

* To use this code, you can get gpu objects that contains gpu information.
* You can set GPU backend by using this function.

### GET CURRENT STATE
##### Defualt
```python3
from tf_alloc import current
current(
    gpu_id = False, 
    total_memory=False, 
    used = False, 
    free = False, 
    percentage_of_use = False,
    percentage_of_free = False,
)
```

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


# 한국어는 간단하게!
## 설치
    pip install tf-alloc

## 문제정의:
* 텐서플로우는 파이토치와 다르게 훈련시 GPU를 전부 할당해버립니다.
* 그러나 실제로 GPU를 모두 사용하지 않기 때문에 큰 낭비가 발생합니다.
* 이를 막기 위해 두가지 방법이 사용되는데
1. GPU를 1개만 쓰도록 제한하기
2. GPU에서 특정 메모리만큼만 사용하도록 제한하기

* 이 두가지 입니다. 그러나 이 방법을 위해선 복잡한 코드와 메모리 관리가 필요합니다.

## 해결책:
* 이것을 해결하기 위해 자동으로 몇번 GPU를 얼만큼만 할당할지 정해주는 코드를 만들었습니다.
* 함수 하나만 사용하면 됩니다.
```python
# On the top of the code
from tf_alloc import allocate as talloc
talloc(gpu=1, percentage=0.5)

import tensorflow as tf
""" your code"""
```
* 맨위에 tf_alloc에서 allocate함수를 불러다가 gpu파라미터와 percentage 파라미터를 주어 호출합니다.
* 그러면 자동으로 몇번의 GPU를 얼만큼의 비율로 사용할지 정해서 할당합니다.
* 매우 쉽습니다.

### 파라미터 설명
* gpu = 몇범 GPU를 쓸 것인지 GPU의 아이디를 넣어줍니다. (만약 gpu가 2개 있다면 0, 1 이 아이디가 됩니다.)
* percentage = 선택한 GPU를 몇의 비율로 쓸건지 정해줍니다. (1.0을 넣으면 해당 GPU를 전부 씁니다)

* 만약 percentage가 몇인지 모른다면 0에서 1 사이의 값을 넣어서 할당해보면 최대 사용가능량이 얼만큼이라고 에러를 출력하니까 걱정없이 사용하시면 됩니다. 다른 훈련에 방해를 주지 않기 때문에, nvidia-smi를 쳐가면서 할당을 하는 것보다 매우 안정적입니다.

* 핵심기능만 한국어로 써 놓았고, 다른 기능은 영문버전을 확인해보시면 감사하겠습니다.
