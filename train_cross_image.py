"""
Training script for SALAD with Cross-Image Learning.
Loads pretrained SALADBase weights and trains CrossImageEncoder.
"""
import argparse
import pytorch_lightning as pl
import torch

from vpr_model import VPRModel
from dataloaders.GSVCitiesDataloader import GSVCitiesDataModule


def main(args):
    # DataModule
    datamodule = GSVCitiesDataModule(
        batch_size=args.batch_size,
        img_per_place=args.img_per_place,
        min_img_per_place=args.img_per_place,
        shuffle_all=False,
        random_sample_from_each_place=True,
        image_size=(args.image_size, args.image_size),
        num_workers=args.num_workers,
        show_data_stats=True,
        val_set_names=args.val_sets.split(","),
    )

    # Model
    model = VPRModel(
        backbone_arch="dinov2_vitb14",
        backbone_config={
            "num_trainable_blocks": args.num_trainable_blocks,
            "return_token": True,
            "norm_layer": True,
        },
        agg_arch="SALAD",
        agg_config={
            "num_channels": 768,
            "num_clusters": 64,
            "cluster_dim": 128,
            "token_dim": 256,
            "img_per_place": args.img_per_place,
        },
        lr=args.lr,
        optimizer="adamw",
        weight_decay=args.weight_decay,
        momentum=0.9,
        lr_sched="linear",
        lr_sched_args={
            "start_factor": 1,
            "end_factor": 0.2,
            "total_iters": args.total_iters,
        },
        loss_name=args.loss,
        miner_name=args.miner,
        miner_margin=args.miner_margin,
        faiss_gpu=args.faiss_gpu,
    )

    # Load pretrained weights
    if args.pretrained_path:
        print(f"\n=== Loading pretrained weights from {args.pretrained_path} ===")
        model.aggregator.load_base_weights(args.pretrained_path, strict=False)

        if args.freeze_base:
            print("=== Freezing SALADBase, only training CrossImageEncoder ===\n")
            model.aggregator.freeze_base()

    # Checkpoint callback
    checkpoint_cb = pl.callbacks.ModelCheckpoint(
        monitor=f"{args.val_sets.split(',')[0]}/R1",
        filename=f"{model.encoder_arch}_cross_image"
        + "_({epoch:02d})_R1[{msls_val/R1:.4f}]_R5[{msls_val/R5:.4f}]",
        auto_insert_metric_name=False,
        save_weights_only=True,
        save_top_k=3,
        save_last=True,
        mode="max",
    )

    # Trainer
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.devices,
        default_root_dir=args.log_dir,
        num_nodes=1,
        num_sanity_val_steps=1,
        precision="16-mixed",
        max_epochs=args.epochs,
        check_val_every_n_epoch=1,
        callbacks=[checkpoint_cb],
        reload_dataloaders_every_n_epochs=1,
        log_every_n_steps=20,
    )

    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SALAD with Cross-Image Learning")

    # Pretrained weights
    parser.add_argument(
        "--pretrained_path",
        type=str,
        default="pretrainedWeight/Salad/last.ckpt",
        help="Path to pretrained SALAD checkpoint",
    )
    parser.add_argument(
        "--freeze_base",
        action="store_true",
        help="Freeze SALADBase, only train CrossImageEncoder",
    )

    # Data
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--img_per_place", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--val_sets", type=str, default="msls_val")

    # Model
    parser.add_argument("--num_trainable_blocks", type=int, default=4)

    # Training
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=6e-5)
    parser.add_argument("--weight_decay", type=float, default=9.5e-9)
    parser.add_argument("--total_iters", type=int, default=4000)

    # Loss
    parser.add_argument("--loss", type=str, default="MultiSimilarityLoss")
    parser.add_argument("--miner", type=str, default="MultiSimilarityMiner")
    parser.add_argument("--miner_margin", type=float, default=0.1)

    # Hardware
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--faiss_gpu", action="store_true")
    parser.add_argument("--log_dir", type=str, default="./logs/cross_image/")

    args = parser.parse_args()
    main(args)
