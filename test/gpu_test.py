import torch

print("="*60)
print("PyTorch CUDA Configuration")
print("="*60)
print(f"PyTorch version:      {torch.__version__}")
print(f"PyTorch CUDA version: {torch.version.cuda}")
print(f"CUDA available:       {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA device count:    {torch.cuda.device_count()}")
    print(f"Current CUDA device:  {torch.cuda.current_device()}")
    print(f"Device name:          {torch.cuda.get_device_name(0)}")
    print(f"Device capability:    {torch.cuda.get_device_capability(0)}")
    
    # Quick performance test
    print("\nTesting GPU computation...")
    x = torch.randn(1000, 1000, device='cuda')
    y = torch.randn(1000, 1000, device='cuda')
    z = torch.matmul(x, y)
    print(f"✓ Success! Computed {z.shape} matrix on GPU")
else:
    print("\n⚠ WARNING: CUDA not available!")
