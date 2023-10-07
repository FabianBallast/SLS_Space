import nvidia_smi

nvidia_smi.nvmlInit()

for i in range(4):
    print(f"GPU {i+1}:")
    
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    print(f"Total memory: {info.total:11}")
    print(f"Free memory:  {info.free:11}")
    print(f"Used memory:  {info.used:11}")
    
    if i < 3:
        print()

nvidia_smi.nvmlShutdown()
