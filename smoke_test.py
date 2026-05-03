import sys
print('Testing PyTorch stack...')
import torch, torchvision, torchaudio
print(f'  torch: {torch.__version__}')
print(f'  CUDA: {torch.version.cuda}')
print(f'  cuDNN: {torch.backends.cudnn.version()}')
print(f'  CUDA available: {torch.cuda.is_available()}')

print('Testing torchaudio extension...')
import torchaudio._extension
print('  ✓ torchaudio._extension loads')

print('Testing MMCV stack...')
import mmcv
print(f'  mmcv: {mmcv.__version__}')

try:
    import mmcv._ext
    print('  ✓ mmcv._ext loads (CUDA ops available)')
except ImportError as e:
    print(f'  ⚠ mmcv._ext unavailable: {e}')
    print('  (This is OK if INSTALL_MMPOSE=0)')

try:
    import mmdet, mmpose
    print(f'  mmdet: {mmdet.__version__}')
    print(f'  mmpose: {mmpose.__version__}')
except ImportError:
    print('  ⚠ mmdet/mmpose not installed (INSTALL_MMPOSE=0)')

print('\\n=== ALL CRITICAL IMPORTS SUCCESSFUL ===')
