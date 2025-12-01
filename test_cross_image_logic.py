"""
Test script to verify cross-image logic is correct.
"""
import torch
from models.aggregators import SALAD


def test_duplicate_order():
    """Test that duplicate order is correct for cross-image grouping."""
    print("\n" + "="*60)
    print(" Testing Duplicate Order")
    print("="*60)
    
    salad = SALAD(num_channels=768, num_clusters=64, cluster_dim=128, token_dim=256, img_per_place=4)
    
    # Create dummy input with distinct values
    B = 3  # 3 images
    features = torch.arange(B).float().view(B, 1, 1, 1).expand(B, 768, 16, 16)
    token = torch.arange(B).float().view(B, 1).expand(B, 768)
    
    # Test repeat_interleave
    features_dup = features.repeat_interleave(4, dim=0)
    
    print(f"\nOriginal batch size: {B}")
    print(f"Duplicated batch size: {features_dup.shape[0]}")
    
    # Check grouping
    print("\nGrouping check (should be [0,0,0,0, 1,1,1,1, 2,2,2,2]):")
    print(features_dup[:, 0, 0, 0].tolist())
    
    expected = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
    actual = features_dup[:, 0, 0, 0].tolist()
    
    if actual == expected:
        print("✓ Grouping is correct!")
    else:
        print("✗ Grouping is WRONG!")
        print(f"  Expected: {expected}")
        print(f"  Actual: {actual}")


def test_cross_image_consistency():
    """Test that cross-image output is consistent for duplicated inputs."""
    print("\n" + "="*60)
    print(" Testing Cross-Image Consistency")
    print("="*60)
    
    salad = SALAD(num_channels=768, num_clusters=64, cluster_dim=128, token_dim=256, img_per_place=4)
    salad.eval()
    
    torch.manual_seed(42)
    
    # Single image
    features = torch.randn(1, 768, 16, 16)
    token = torch.randn(1, 768)
    
    with torch.no_grad():
        # Method 1: forward_single_with_cross_image
        desc1 = salad.forward_single_with_cross_image((features, token))
        
        # Method 2: Manual duplicate and forward
        features_dup = features.repeat_interleave(4, dim=0)
        token_dup = token.repeat_interleave(4, dim=0)
        s, t = salad.base.compute_features((features_dup, token_dup))
        s = salad.cross_encoder(s)
        desc_all = salad.base.build_descriptor(s, t)
        
    print(f"\nDescriptor from forward_single_with_cross_image: {desc1.shape}")
    print(f"All 4 duplicates descriptors: {desc_all.shape}")
    
    # Check if all 4 duplicates give same descriptor
    print("\nChecking if all 4 duplicates give same descriptor:")
    for i in range(4):
        diff = (desc_all[0] - desc_all[i]).abs().max().item()
        print(f"  Diff between copy 0 and copy {i}: {diff:.6f}")
    
    # Check if forward_single_with_cross_image matches
    diff_method = (desc1[0] - desc_all[0]).abs().max().item()
    print(f"\nDiff between forward_single_with_cross_image and manual: {diff_method:.6f}")


def test_training_vs_validation():
    """Test that training and validation use same distribution."""
    print("\n" + "="*60)
    print(" Testing Training vs Validation Distribution")
    print("="*60)
    
    salad = SALAD(num_channels=768, num_clusters=64, cluster_dim=128, token_dim=256, img_per_place=4)
    
    torch.manual_seed(42)
    
    # Simulate training: 4 different images from same place
    features_train = torch.randn(4, 768, 16, 16)
    token_train = torch.randn(4, 768)
    
    # Simulate validation: 1 image duplicated 4 times
    features_val = features_train[0:1]  # Take first image
    token_val = token_train[0:1]
    
    salad.train()
    desc_train = salad((features_train, token_train))
    
    salad.eval()
    with torch.no_grad():
        desc_val = salad.forward_single_with_cross_image((features_val, token_val))
    
    print(f"\nTraining descriptor (4 different images): {desc_train.shape}")
    print(f"Validation descriptor (1 image duplicated): {desc_val.shape}")
    
    # Compare first training descriptor with validation
    cos_sim = torch.nn.functional.cosine_similarity(desc_train[0:1], desc_val, dim=-1)
    print(f"\nCosine similarity between train[0] and val: {cos_sim.item():.4f}")
    
    print("\n" + "-"*40)
    print("Analysis:")
    print("-"*40)
    print("""
Training: 4 DIFFERENT images → cross-image learns from diversity
Validation: 4 IDENTICAL images → cross-image sees no diversity

This is a fundamental mismatch!

The cross-image encoder learns to combine information from
different viewpoints, but at inference time, it only sees
the same image repeated.

Possible solutions:
1. Don't use cross-image (baseline SALAD)
2. Use cross-image only for database (where you have multiple images)
3. Redesign cross-image to work with single images (self-attention on patches)
""")


def test_output_norm():
    """Test that output is properly normalized."""
    print("\n" + "="*60)
    print(" Testing Output Normalization")
    print("="*60)
    
    salad = SALAD(num_channels=768, num_clusters=64, cluster_dim=128, token_dim=256, img_per_place=4)
    salad.eval()
    
    torch.manual_seed(42)
    features = torch.randn(2, 768, 16, 16)
    token = torch.randn(2, 768)
    
    with torch.no_grad():
        desc = salad.forward_single_with_cross_image((features, token))
    
    norms = desc.norm(dim=-1)
    print(f"\nDescriptor norms (should be ~1.0): {norms.tolist()}")
    
    if all(abs(n - 1.0) < 0.01 for n in norms.tolist()):
        print("✓ Normalization is correct!")
    else:
        print("✗ Normalization issue!")


if __name__ == "__main__":
    test_duplicate_order()
    test_cross_image_consistency()
    test_training_vs_validation()
    test_output_norm()
    
    print("\n" + "="*60)
    print(" Summary")
    print("="*60)
    print("""
The fundamental issue is:
- Training: Cross-image learns from 4 DIFFERENT images
- Validation: Cross-image sees 4 IDENTICAL images

This creates a distribution mismatch that may hurt performance.

Recommendation: Test both approaches:
1. With cross-image duplication (current)
2. Without cross-image (baseline)

Compare results to see if cross-image actually helps.
""")
