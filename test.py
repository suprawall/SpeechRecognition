import torch

print("CUDA disponible :", torch.cuda.is_available())
print("Nombre de GPUs disponibles :", torch.cuda.device_count())