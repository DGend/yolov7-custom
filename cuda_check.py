import torch

def calculate_max_batch_size(model, device):
    batch_size = 1
    while True:
        try:
            x = torch.randn(batch_size, 3, 224, 224).to(device)
            model(x)
            batch_size *= 2
        except RuntimeError:
            break
    return batch_size // 2

if torch.cuda.is_available():
    print("CUDA is available")
    device = torch.device("cuda")
    x = torch.tensor([1.0, 2.0, 3.0]).to(device)
    print("Tensor created on GPU:", x)
    model = YourModel()  # Replace YourModel with your actual model
    max_batch_size = calculate_max_batch_size(model, device)
    print("Max batch size:", max_batch_size)
else:
    print("CUDA is not available")

