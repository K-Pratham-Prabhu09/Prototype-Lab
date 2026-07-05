import torch
import time

# Create a massive 10,000 x 10,000 matrix
size = 25000

# --- CPU TEST ---
print("Running on CPU...")
cpu_matrix = torch.randn(size, size)

start_cpu = time.time()
_ = torch.matmul(cpu_matrix, cpu_matrix)
cpu_time = time.time() - start_cpu
print(f"CPU Time: {cpu_time:.4f} seconds")

# --- GPU TEST ---
if torch.cuda.is_available():
    print("\nRunning on GPU (RTX 3050)...")
    gpu_matrix = torch.randn(size, size, device="cuda")
    
    # WARM-UP (GPU needs a dummy run to wake up/allocate memory)
    _ = torch.matmul(gpu_matrix, gpu_matrix)
    torch.cuda.synchronize() 
    
    start_gpu = time.time()
    _ = torch.matmul(gpu_matrix, gpu_matrix)
    torch.cuda.synchronize() # Forces Python to wait for the GPU to finish
    gpu_time = time.time() - start_gpu
    
    print(f"GPU Time: {gpu_time:.4f} seconds")
    print(f"\nResult: Your GPU is {cpu_time / gpu_time:.2f}x faster than your CPU!")
else:
    print("CUDA is not available. Check your PyTorch installation!")