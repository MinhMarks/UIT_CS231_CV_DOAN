"""
Debug script to check model behavior and weight loading.
"""
import torch
from pathlib import Path


def check_pretrained_weights():
    """Check what's in the pretrained checkpoint."""
    print("\n" + "="*60)
    print(" Checking Pretrained Weights")
    print("="*60)
    
    ckpt_path = "pretrainedWeight/Salad/last.ckpt"
    if not Path(ckpt_path).exists():
        print(f"Checkpoint not found: {ckpt_path}")
        return
    
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    
    print(f"\nTotal keys in checkpoint: {len(state_dict)}")
    print("\nKeys (first 20):")
    for i, k in enumerate(list(state_dict.keys())[:20]):
        print(f"  {k}: {state_dict[k].shape}")
    
    # Check for aggregator keys
    agg_keys = [k for k in state_dict.keys() if "aggregator" in k.lower()]
    print(f"\nAggregator keys: {len(agg_keys)}")
    for k in agg_keys[:10]:
        print(f"  {k}: {state_dict[k].shape}")
    
    return state_dict


def compare_weights():
    """Compare weights before and after loading pretrained."""
    print("\n" + "="*60)
    print(" Comparing Weights Before/After Loading")
    print("="*60)
    
    from models.aggregators import SALAD
    
    # Create fresh model
    model1 = SALAD(num_channels=768, num_clusters=64, cluster_dim=128, token_dim=256)
    
    # Get initial weights
    initial_weights = {k: v.clone() for k, v in model1.base.state_dict().items()}
    
    # Load pretrained
    ckpt_path = "pretrainedWeight/Salad/last.ckpt"
    if Path(ckpt_path).exists():
        model1.load_base_weights(ckpt_path, strict=False)
    
    # Compare
    loaded_weights = model1.base.state_dict()
    
    print("\nWeight comparison:")
    changed = 0
    unchanged = 0
    for k in initial_weights:
        if torch.allclose(initial_weights[k], loaded_weights[k]):
            unchanged += 1
        else:
            changed += 1
            print(f"  CHANGED: {k}")
    
    print(f"\nChanged: {changed}, Unchanged: {unchanged}")
    
    if changed == 0:
        print("\n⚠️  WARNING: No weights were loaded! Check key mapping.")


def test_output_consistency():
    """Test if outputs are consistent between modes."""
    print("\n" + "="*60)
    print(" Testing Output Consistency")
    print("="*60)
    
    from models.aggregators import SALAD
    
    model = SALAD(num_channels=768, num_clusters=64, cluster_dim=128, token_dim=256, img_per_place=4)
    
    # Load pretrained if available
    ckpt_path = "pretrainedWeight/Salad/last.ckpt"
    if Path(ckpt_path).exists():
        model.load_base_weights(ckpt_path, strict=False)
    
    # Create dummy input
    torch.manual_seed(42)
    x = torch.randn(8, 768, 16, 16)  # 8 images = 2 places x 4 images
    t = torch.randn(8, 768)
    
    # Test 1: Training mode with cross-image
    model.train()
    out_train = model((x, t))
    print(f"\nTraining mode output: {out_train.shape}")
    print(f"  Mean: {out_train.mean().item():.6f}")
    print(f"  Std: {out_train.std().item():.6f}")
    print(f"  Norm: {out_train.norm(dim=-1).mean().item():.6f}")
    
    # Test 2: Eval mode (no cross-image)
    model.eval()
    with torch.no_grad():
        out_eval = model((x, t))
    print(f"\nEval mode output: {out_eval.shape}")
    print(f"  Mean: {out_eval.mean().item():.6f}")
    print(f"  Std: {out_eval.std().item():.6f}")
    print(f"  Norm: {out_eval.norm(dim=-1).mean().item():.6f}")
    
    # Test 3: forward_single
    with torch.no_grad():
        out_single = model.forward_single((x, t))
    print(f"\nforward_single output: {out_single.shape}")
    print(f"  Mean: {out_single.mean().item():.6f}")
    print(f"  Std: {out_single.std().item():.6f}")
    
    # Compare
    diff_train_eval = (out_train - out_eval).abs().mean().item()
    diff_eval_single = (out_eval - out_single).abs().mean().item()
    
    print(f"\nDifference train vs eval: {diff_train_eval:.6f}")
    print(f"Difference eval vs single: {diff_eval_single:.6f}")
    
    if diff_train_eval > 0.01:
        print("\n⚠️  WARNING: Large difference between train and eval outputs!")
        print("   This means cross-image is changing the features significantly.")
        print("   But validation doesn't use cross-image, causing mismatch.")


def test_base_only():
    """Test SALADBase alone (without cross-image)."""
    print("\n" + "="*60)
    print(" Testing SALADBase Only")
    print("="*60)
    
    from models.aggregators import SALADBase
    
    model = SALADBase(num_channels=768, num_clusters=64, cluster_dim=128, token_dim=256)
    
    # Load pretrained
    ckpt_path = "pretrainedWeight/Salad/last.ckpt"
    if Path(ckpt_path).exists():
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        
        # Filter keys
        base_state_dict = {}
        for k, v in state_dict.items():
            new_key = k
            for prefix in ["aggregator.", "model.aggregator.", "base."]:
                if k.startswith(prefix):
                    new_key = k[len(prefix):]
                    break
            
            if any(new_key.startswith(name) for name in ["token_features", "cluster_features", "score", "dust_bin"]):
                base_state_dict[new_key] = v
        
        model.load_state_dict(base_state_dict, strict=False)
        print(f"Loaded {len(base_state_dict)} weights")
    
    # Test
    torch.manual_seed(42)
    x = torch.randn(4, 768, 16, 16)
    t = torch.randn(4, 768)
    
    model.eval()
    with torch.no_grad():
        out = model((x, t))
    
    print(f"\nSALADBase output: {out.shape}")
    print(f"  Mean: {out.mean().item():.6f}")
    print(f"  Std: {out.std().item():.6f}")
    print(f"  Norm: {out.norm(dim=-1).mean().item():.6f}")


def main():
    check_pretrained_weights()
    compare_weights()
    test_output_consistency()
    test_base_only()
    
    print("\n" + "="*60)
    print(" Debug Complete")
    print("="*60)
    print("\nIf weights are loading correctly but results are still low,")
    print("the issue is likely the train-test mismatch from cross-image.")
    print("\nRecommendation: Train WITHOUT cross-image first to establish baseline,")
    print("then compare with cross-image version.")


if __name__ == "__main__":
    main()
