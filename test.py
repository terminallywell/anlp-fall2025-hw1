import torch
tensor_on_cpu = torch.randn(2, 3)
print(tensor_on_cpu.device)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if torch.cuda.is_available():
    tensor_on_gpu = torch.randn(2, 3).cuda()
    print(tensor_on_gpu.device)

torch.tensor([[101, 7592, 2088, 102, 0, 0, 0, 0],
              [101, 7592, 15756, 2897, 2005, 17953, 2361, 102]])
