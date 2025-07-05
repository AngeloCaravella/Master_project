
import torch
print(f"Versione PyTorch: {torch.__version__}")
print(f"CUDA disponibile: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Versione CUDA di PyTorch: {torch.version.cuda}")
    print(f"Numero di GPU: {torch.cuda.device_count()}")
    print(f"GPU corrente: {torch.cuda.current_device()}")
    print(f"Nome GPU: {torch.cuda.get_device_name(0)}")
else:
    print("PyTorch non è in grado di utilizzare la GPU. Potrebbe essere installata la versione solo CPU.")
