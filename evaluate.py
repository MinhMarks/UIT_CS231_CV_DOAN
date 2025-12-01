"""
Evaluation script for SALAD with Cross-Image Learning.
Evaluate trained model on validation/test datasets.
"""
import argparse
import torch
import pytorch_lightning as pl
from pathlib import Path

from vpr_model import VPRModel
from dataloaders.GSVCitiesDataloader import GSVCitiesDataModule


def main(args):
    # DataModule (only for validation)
    datamodule = GSVCitiesDataModule(
        batch_size=args.batch_size,
        img_per_place=4,
        min_img_per_place=4,
        shuffle_all=False,
        random_sample_from_each_place=True,
        image_size=(args.image_size, args.image_size),
        num_workers=args.num_workers,
        show_data_stats=True,
        val_set_names=args.val_sets.split(","),
    )

    # Load model from checkpoint
    print(f"\n=== Loading model from {args.checkpoint} ===\n")

    if args.checkpoint.endswith(".ckpt"):
        # PyTorch Lightning checkpoint
        model = VPRModel.load_from_checkpoint(
            args.checkpoint,
            faiss_gpu=args.faiss_gpu,
        )
    else:
        # Manual loading
        model = VPRModel(
            backbone_arch="dinov2_vitb14",
            backbone_config={
                "num_trainable_blocks": 4,
                "return_token": True,
                "norm_layer": True,
            },
            agg_arch="SALAD",
            agg_config={
                "num_channels": 768,
                "num_clusters": 64,
                "cluster_dim": 128,
                "token_dim": 256,
                "img_per_place": 4,
            },
            faiss_gpu=args.faiss_gpu,
        )
        state_dict = torch.load(args.checkpoint, map_location="cpu")
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        model.load_state_dict(state_dict, strict=False)

    # Trainer for validation only
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="16-mixed" if torch.cuda.is_available() else 32,
        logger=False,
    )

    # Run validation
    print("\n=== Running Evaluation ===\n")
    trainer.validate(model=model, datamodule=datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SALAD model")

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.ckpt or .pth)",
    )
    parser.add_argument(
        "--val_sets",
        type=str,
        default="msls_val",
        help="Comma-separated validation sets (msls_val, pitts30k_val, pitts30k_test)",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--faiss_gpu", action="store_true")

    args = parser.parse_args()
    main(args)
