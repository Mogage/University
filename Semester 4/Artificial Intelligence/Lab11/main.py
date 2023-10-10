# from emojis import run
# from emotionsPre import run
# from mainClassifier import run
from multiLabel import run

from tensorflow.python.client import device_lib

if __name__ == "__main__":
    import torch

    torch.cuda.empty_cache()
    print(device_lib.list_local_devices())
    # run()
