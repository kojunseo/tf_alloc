import os
from subprocess import Popen, PIPE
from .utils import safeFloatCast
import tensorflow as tf 


class GPU:
    def __init__(self, **kwargs):
        self.device_id = kwargs["device_id"]
        self.total_memory = kwargs["total_memory"]
        self.used_memory = kwargs["used_memory"]
        self.freed_memory = kwargs["freed_memory"]
        self.gpu_name = kwargs["gpu_name"]

    def __repr__(self):
        return f"GPU ID: {self.device_id} || Total Memory: {self.total_memory} || Used: {self.used_memory} || Free: {self.freed_memory} || Percent of use: {round(self.used_memory/self.total_memory, 2)}"

    def state(
            self, 
            gpu_id = True, 
            total_memory=True, 
            used = True, 
            free = True, 
            percentage_of_use = True,
            percentage_of_free = True
        ):
        output_list = []

        if gpu_id:
            output_list.append(f"GPU ID: {self.device_id}")
        if total_memory:
            output_list.append(f"Total Memory: {self.total_memory}")
        if used:
            output_list.append(f"Used: {self.used_memory}")
        if free:
            output_list.append(f"Free: {self.freed_memory}")

        if percentage_of_use:
            output_list.append(f"Percentage of Use: {round(self.used_memory/self.total_memory, 2)}")
        if percentage_of_free:
            output_list.append(f"Percentage of Free: {round(self.freed_memory/self.total_memory, 2)}")
        return " || ".join(output_list)


    def allocate(self, percentage):
        if percentage == 0:
            raise ValueError("You cannot allocate 0 percentage")
        if type(percentage) != float:
            raise TypeError("percentage parameter must be float, if you want to allocate full gpu, then use percentage = 1.0")
        if percentage>1 or percentage <0:
            raise ValueError("percentage parameter must be between 0 and 1")

        maximum_allocation = round(self.freed_memory/self.total_memory, 2)
        if percentage == 1.0 and maximum_allocation == 1.0:
            if self.used_memory/self.total_memory > 1:
                raise ValueError(f"GPU memory is currently allocated to the other process(es). Terminate process on this GPU first. Maximun allocation can be {maximum_allocation}")
            else:
                os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
                os.environ["CUDA_VISIBLE_DEVICES"]=str(self.device_id)

        else:
            if percentage > maximum_allocation*0.9:
                raise ValueError(f"You cannot allocate GPU with {percentage}. Possible maximun allocation is {round(maximum_allocation*0.9,3)}")
            else:
                os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
                os.environ["CUDA_VISIBLE_DEVICES"]=str(self.device_id)
                if tf.__version__.startswith("2"):
                    TF_GPU_OBJECTs = tf.config.experimental.list_physical_devices('GPU')
                    if TF_GPU_OBJECTs:
                        tf.config.experimental.set_virtual_device_configuration(
                            TF_GPU_OBJECTs[0], 
                            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=((self.total_memory) * percentage*0.9))]
                        )

                elif tf.__version__.startswith("1"):
                    config = tf.ConfigProto()
                    config.gpu_options.per_process_gpu_memory_fraction = percentage
                    config.gpu_options.allow_growth = False
                    session = tf.Session(config=config)
                    return session

                else:
                    ver = tf.__version__[0]
                    raise ImportError(f"Your Tensorflow version is {ver}. This is not a official tensorflow version name")
                

        
class GPUs:
    def __init__(self):
        self.num_gpu = 0
        self.gpus = [] 

    def append(self, gpu):
        self.gpus.append(gpu)
        self.num_gpu += 1

    def __getitem__(self,index):
        for g in self.gpus:
            if g.device_id == index:
                return g

        raise IndexError(f"Device index out of range")

    def __indiv__(self,gpu_id,total_memory,used,free,percentage_of_use,percentage_of_free):
        output_str_format = "\t{}\n"
        output = ""
        for g in self.gpus:
            output += output_str_format.format(g.state(
               gpu_id,total_memory,used,free,percentage_of_use,percentage_of_free
            ))
        return output
    
    def __len__(self):
        return len(self.gpus)

    def state(self,gpu_id,total_memory,used,free,percentage_of_use,percentage_of_free) -> str:
        if self.num_gpu == 0:
            return "There is no GPU in this device"

        return f"Num GPUs: {self.num_gpu}\nGPU Stats:\n{self.__indiv__(gpu_id,total_memory,used,free,percentage_of_use,percentage_of_free)}"

def get_gpu_objects():
    GPU_IN_MACHINE = GPUs()

    nvidia_smi = "nvidia-smi"    
    try:
        p = Popen([nvidia_smi,"--query-gpu=index,uuid,utilization.gpu,memory.total,memory.used,memory.free,driver_version,name,gpu_serial,display_active,display_mode,temperature.gpu", "--format=csv,noheader,nounits"], stdout=PIPE)
    except:
        raise SystemError("nvidia driver is not detected")
    stdout, _ = p.communicate()
    output = stdout.decode('UTF-8')

    for g in output.split(os.linesep):
        if len(g) == 0:
            continue
        dev_info = g.split(", ")
        gpu = GPU(
            device_id= int(dev_info[0]),
            total_memory= safeFloatCast(dev_info[3]),
            used_memory= safeFloatCast(dev_info[4]),
            freed_memory=safeFloatCast(dev_info[5]),
            gpu_name= dev_info[7]
        )
        GPU_IN_MACHINE.append(gpu)

    return GPU_IN_MACHINE


def current(
        gpu_id = False, 
        total_memory=False, 
        used = False, 
        free = False, 
        percentage_of_use = False,
        percentage_of_free = False
    ):
    gpu_objs = get_gpu_objects()
    if (
        gpu_id == False and
        total_memory == False and
        used == False and
        free == False and
        percentage_of_use == False and
        percentage_of_free == False
    ):
        if len(gpu_objs)==0:
            print("No GPU")
            return
        else:
            for g in gpu_objs:
                print(f"The maximum allocation is {round(g.freed_memory / g.total_memory,2)*0.9}, for safety")
                return

    print(gpu_objs.state(gpu_id,total_memory,used,free,percentage_of_use,percentage_of_free))
    for g in gpu_objs:
        print(f"The maximum allocation for GPU{g.device_id} is {round(g.freed_memory / g.total_memory,2)*0.9}, for safety")

# if __name__ == "__main__":
#     GPUs = get_gpu_in_machine()
#     GPUs["gpus"][0].allocate_gpu_function(1.0)
