# RTX 50 系 (sm_120) 需 PyTorch CUDA 12.8+，在 myenv 中执行:
#   .\install_gpu_torch.ps1

$Python = "D:\anaconda\envs\myenv\python.exe"
if (-not (Test-Path $Python)) { $Python = "python" }

& $Python -m pip uninstall torch torchvision -y
& $Python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 --no-cache-dir
& $Python -c "import torch; x=torch.randn(2,3,device='cuda'); print(torch.__version__, torch.cuda.get_device_name(0), 'OK')"
