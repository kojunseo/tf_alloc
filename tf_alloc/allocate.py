from .gpu import get_gpu_objects

def allocate(gpu, percentage):
    GPUs = get_gpu_objects()
    if len(GPUs) == 0:
        assert SystemError("There is no detected GPU on the devices with our library.")

    else:
        GPUs[gpu].allocate(percentage)