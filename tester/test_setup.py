"""
Test script to verify all connections, imports, and paths are working correctly.
Run this before training to catch any issues early.
"""
import sys
from pathlib import Path


def print_header(title):
    print(f"\n{'='*60}")
    print(f" {title}")
    print('='*60)


def print_ok(msg):
    print(f"  ✓ {msg}")


def print_fail(msg):
    print(f"  ✗ {msg}")


def print_warn(msg):
    print(f"  ⚠ {msg}")


def test_imports():
    """Test all required imports."""
    print_header("Testing Imports")
    
    errors = []
    
    # Core
    try:
        import torch
        print_ok(f"PyTorch {torch.__version__}")
    except ImportError as e:
        print_fail(f"PyTorch: {e}")
        errors.append("torch")

    try:
        import pytorch_lightning as pl
        print_ok(f"PyTorch Lightning {pl.__version__}")
    except ImportError as e:
        print_fail(f"PyTorch Lightning: {e}")
        errors.append("pytorch_lightning")

    # Models
    try:
        from models.aggregators import SALAD, SALADBase, CrossImageEncoder
        print_ok("SALAD, SALADBase, CrossImageEncoder")
    except ImportError as e:
        print_fail(f"Aggregators: {e}")
        errors.append("aggregators")

    try:
        from models import helper
        print_ok("models.helper")
    except ImportError as e:
        print_fail(f"models.helper: {e}")
        errors.append("helper")

    # Dataloaders
    try:
        from dataloaders.GSVCitiesDataloader import GSVCitiesDataModule
        print_ok("GSVCitiesDataModule")
    except ImportError as e:
        print_fail(f"GSVCitiesDataloader: {e}")
        errors.append("dataloader")

    # Utils
    try:
        from utils import get_loss, get_miner, get_validation_recalls
        print_ok("utils (loss, miner, validation)")
    except ImportError as e:
        print_fail(f"utils: {e}")
        errors.append("utils")

    # VPR Model
    try:
        from vpr_model import VPRModel
        print_ok("VPRModel")
    except ImportError as e:
        print_fail(f"VPRModel: {e}")
        errors.append("vpr_model")

    return len(errors) == 0


def test_paths():
    """Test all required paths exist."""
    print_header("Testing Paths")
    
    paths_to_check = [
        ("models/aggregators/salad.py", True),
        ("models/aggregators/salad_base.py", True),
        ("models/aggregators/cross_image_encoder.py", True),
        ("pretrainedWeight/Salad/last.ckpt", True),
        ("datasets/msls_val/", False),
        ("datasets/gsv-cities/", False),
    ]
    
    all_ok = True
    for path, required in paths_to_check:
        p = Path(path)
        exists = p.exists()
        
        if exists:
            print_ok(f"{path}")
        elif required:
            print_fail(f"{path} (REQUIRED)")
            all_ok = False
        else:
            print_warn(f"{path} (optional, needed for training/validation)")
    
    return all_ok


def test_model_creation():
    """Test model can be created."""
    print_header("Testing Model Creation")
    
    try:
        import torch
        from models.aggregators import SALAD, SALADBase, CrossImageEncoder
        
        # Test SALADBase
        base = SALADBase(num_channels=768, num_clusters=64, cluster_dim=128, token_dim=256)
        print_ok(f"SALADBase created: {sum(p.numel() for p in base.parameters()):,} params")
        
        # Test CrossImageEncoder
        cross = CrossImageEncoder(cluster_dim=128, num_clusters=64, img_per_place=4)
        print_ok(f"CrossImageEncoder created: {sum(p.numel() for p in cross.parameters()):,} params")
        
        # Test SALAD
        salad = SALAD(num_channels=768, num_clusters=64, cluster_dim=128, token_dim=256, img_per_place=4)
        print_ok(f"SALAD created: {sum(p.numel() for p in salad.parameters()):,} params")
        
        return True
    except Exception as e:
        print_fail(f"Model creation failed: {e}")
        return False


def test_forward_pass():
    """Test forward pass with dummy data."""
    print_header("Testing Forward Pass")
    
    try:
        import torch
        from models.aggregators import SALAD
        
        salad = SALAD(num_channels=768, num_clusters=64, cluster_dim=128, token_dim=256, img_per_place=4)
        
        # Dummy input: (features, token)
        batch_size = 8  # Must be divisible by img_per_place=4
        features = torch.randn(batch_size, 768, 16, 16)
        token = torch.randn(batch_size, 768)
        
        # Test training mode (with cross-image)
        salad.train()
        out_train = salad((features, token))
        expected_dim = 64 * 128 + 256  # num_clusters * cluster_dim + token_dim
        print_ok(f"Training forward: input ({batch_size}, 768, 16, 16) -> output {tuple(out_train.shape)}")
        
        # Test eval mode (single image)
        salad.eval()
        with torch.no_grad():
            out_eval = salad((features, token))
        print_ok(f"Eval forward: input ({batch_size}, 768, 16, 16) -> output {tuple(out_eval.shape)}")
        
        # Test forward_single
        single_features = torch.randn(1, 768, 16, 16)
        single_token = torch.randn(1, 768)
        with torch.no_grad():
            out_single = salad.forward_single((single_features, single_token))
        print_ok(f"Single forward: input (1, 768, 16, 16) -> output {tuple(out_single.shape)}")
        
        # Verify output dimension
        assert out_train.shape[-1] == expected_dim, f"Expected {expected_dim}, got {out_train.shape[-1]}"
        print_ok(f"Output dimension correct: {expected_dim}")
        
        return True
    except Exception as e:
        print_fail(f"Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pretrained_loading():
    """Test loading pretrained weights."""
    print_header("Testing Pretrained Weight Loading")
    
    checkpoint_path = Path("pretrainedWeight/Salad/last.ckpt")
    
    if not checkpoint_path.exists():
        print_warn(f"Checkpoint not found: {checkpoint_path}")
        print_warn("Skipping pretrained loading test")
        return True
    
    try:
        import torch
        from models.aggregators import SALAD
        
        salad = SALAD(num_channels=768, num_clusters=64, cluster_dim=128, token_dim=256, img_per_place=4)
        
        # Load weights
        salad.load_base_weights(str(checkpoint_path), strict=False)
        print_ok("Pretrained weights loaded successfully")
        
        # Test freeze/unfreeze
        salad.freeze_base()
        frozen_params = sum(1 for p in salad.base.parameters() if not p.requires_grad)
        total_base_params = sum(1 for p in salad.base.parameters())
        print_ok(f"Freeze base: {frozen_params}/{total_base_params} params frozen")
        
        salad.unfreeze_base()
        unfrozen_params = sum(1 for p in salad.base.parameters() if p.requires_grad)
        print_ok(f"Unfreeze base: {unfrozen_params}/{total_base_params} params unfrozen")
        
        return True
    except Exception as e:
        print_fail(f"Pretrained loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gpu():
    """Test GPU availability."""
    print_header("Testing GPU")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            print_ok(f"CUDA available: {torch.cuda.get_device_name(0)}")
            print_ok(f"CUDA version: {torch.version.cuda}")
            
            # Test model on GPU
            from models.aggregators import SALAD
            salad = SALAD(num_channels=768).cuda()
            x = (torch.randn(4, 768, 16, 16).cuda(), torch.randn(4, 768).cuda())
            salad.train()
            out = salad(x)
            print_ok(f"GPU forward pass successful")
            
            # Memory info
            mem_allocated = torch.cuda.memory_allocated() / 1024**2
            mem_reserved = torch.cuda.memory_reserved() / 1024**2
            print_ok(f"GPU memory: {mem_allocated:.1f}MB allocated, {mem_reserved:.1f}MB reserved")
        else:
            print_warn("CUDA not available, will use CPU")
        
        return True
    except Exception as e:
        print_fail(f"GPU test failed: {e}")
        return False


def main():
    print("\n" + "="*60)
    print(" SALAD Cross-Image Setup Test")
    print("="*60)
    
    results = {
        "Imports": test_imports(),
        "Paths": test_paths(),
        "Model Creation": test_model_creation(),
        "Forward Pass": test_forward_pass(),
        "Pretrained Loading": test_pretrained_loading(),
        "GPU": test_gpu(),
    }
    
    print_header("Summary")
    
    all_passed = True
    for name, passed in results.items():
        if passed:
            print_ok(name)
        else:
            print_fail(name)
            all_passed = False
    
    print()
    if all_passed:
        print("All tests passed! Ready to train.")
        return 0
    else:
        print("Some tests failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
