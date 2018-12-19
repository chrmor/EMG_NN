
import GPUtil

# Get the first available GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
try:
    deviceIDs = GPUtil.getAvailable(order='memory', limit=1, maxLoad=100, maxMemory=20)  # return a list of available gpus

except:
    print('GPU not compatible with NVIDIA-SMI')

else:
    print(deviceIDs[0])
    os.environ["CUDA_VISIBLE_DEVICES"] = str(deviceIDs[0])