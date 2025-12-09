"""
Debug: Compare baseline SALAD vs Cross-Image SALAD
"""
import torch
from pathlib import Path


def test_pretrained_baseline():
    """Test pretrained SALAD without any cross-image modification."""
    print("\n" + "="*60)
    print(" Testing Pretrained SALAD Baseline")
    print("="*60)
    
    from models.aggregators import SALADBase, SALAD
    
    ckpt_path = "pretrainedWeight/Salad/last.ckpt"
    if not Path(ckpt_path).exists():
        print(f"Checkpoint not found: {ckpt_path}")
        return
    
    # Load checkpoint
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    
    print(f"\nCheckpoint keys containing 'aggregator':")
    agg_keys = [k for k in state_dict.keys() if "aggregator" in k]
    for k in agg_keys:
        print(f"  {k}: {state_dict[k].shape}")
    
    # Create SALADBase and load weights
    base = SALADBase(num_channels=768, num_clusters=64, cluster_dim=128, token_dim=256)
    
    # Map weights
    base_state_dict = {}
    for k, v in state_dict.items():
        if "aggregator." in k:
            new_key = k.replace("aggregator.", "")
            if any(new_key.startswith(name) for name in ["token_features", "cluster_features", "score", "dust_bin"]):
                base_state_dict[new_key] = v
                print(f"  Mapping: {k} -> {new_key}")
    
    # Load
    missing, unexpected = base.load_state_dict(base_state_dict, strict=False)
    print(f"\nLoaded {len(base_state_dict)} weights")
    print(f"Missing keys: {missing}")
    print(f"Unexpected keys: {unexpected}")
    
    # Test forward
    torch.manual_seed(42)
    x = torch.randn(4, 768, 16, 16)
    t = torch.randn(4, 768)
    
    base.eval()
    with torch.no_grad():
        out = base((x, t))
    
    print(f"\nSALADBase output shape: {out.shape}")
    print(f"Output norm (should be ~1.0): {out.norm(dim=-1).mean().item():.4f}")
    
    # Now test full SALAD with cross-image
    print("\n" + "-"*40)
    print("Testing SALAD with cross-image")
    print("-"*40)
    
    salad = SALAD(num_channels=768, num_clusters=64, cluster_dim=128, token_dim=256, img_per_place=4)
    salad.load_base_weights(ckpt_path, strict=False)
    
    # Compare outputs
    salad.eval()
    with torch.no_grad():
        # forward_single should match SALADBase
        out_single = salad.forward_single((x, t))
        
    diff = (out - out_single).abs().max().item()
    print(f"\nDifference between SALADBase and SALAD.forward_single: {diff:.6f}")
    
    if diff < 1e-5:
        print("✓ Outputs match! Pretrained weights loaded correctly.")
    else:
        print("✗ Outputs don't match! Something is wrong with weight loading.")


def check_training_effect():
    """Check how cross-image affects the output during training."""
    print("\n" + "="*60)
    print(" Checking Cross-Image Effect")
    print("="*60)
    
    from models.aggregators import SALAD
    
    ckpt_path = "pretrainedWeight/Salad/last.ckpt"
    
    salad = SALAD(num_channels=768, num_clusters=64, cluster_dim=128, token_dim=256, img_per_place=4)
    
    if Path(ckpt_path).exists():
        salad.load_base_weights(ckpt_path, strict=False)
    
    # Freeze base as in training
    salad.freeze_base()
    
    torch.manual_seed(42)
    x = torch.randn(8, 768, 16, 16)  # 2 places x 4 images
    t = torch.randn(8, 768)
    
    # Eval mode - no cross-image
    salad.eval()
    with torch.no_grad():
        out_eval = salad((x, t))
    
    # Training mode - with cross-image
    salad.train()
    out_train = salad((x, t))
    
    # Compare
    diff = (out_train - out_eval).abs()
    print(f"\nOutput difference (train vs eval):")
    print(f"  Mean diff: {diff.mean().item():.6f}")
    print(f"  Max diff: {diff.max().item():.6f}")
    print(f"  Min diff: {diff.min().item():.6f}")
    
    # Cosine similarity
    cos_sim = torch.nn.functional.cosine_similarity(out_train, out_eval, dim=-1)
    print(f"\nCosine similarity between train and eval outputs:")
    print(f"  Mean: {cos_sim.mean().item():.4f}")
    print(f"  Min: {cos_sim.min().item():.4f}")
    
    if cos_sim.mean() < 0.9:
        print("\n⚠️  WARNING: Cross-image significantly changes the output!")
        print("   This causes train-test mismatch because validation uses eval mode.")
        print("\n   SOLUTION: The cross-image encoder is learning features that")
        print("   don't transfer to single-image inference.")


def suggest_fix():
    print("\n" + "="*60)
    print(" Suggested Fixes")
    print("="*60)
    print("""
1. OPTION A: Don't use cross-image at all
   - Just use SALADBase with pretrained weights
   - This should give you the original SALAD performance

2. OPTION B: Use cross-image ONLY as auxiliary loss
   - Keep the main descriptor from SALADBase
   - Use cross-image for an additional consistency loss
   - Don't modify the actual descriptor

3. OPTION C: Apply cross-image to database at inference time
   - Requires grouping database images by place
   - More complex but maintains consistency

Run this to test baseline performance:
   python evaluate.py --checkpoint pretrainedWeight/Salad/last.ckpt --val_sets msls_val
""")


if __name__ == "__main__":
    test_pretrained_baseline()
    check_training_effect()
    suggest_fix()
